from fastapi.responses import StreamingResponse
import subprocess
import sys


import openai
from dotenv import load_dotenv
import os
import zipfile
from cloud.ImgClass.ImgClassData import ImgClassData
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import uuid
from cloud.aws import AWSHelper

app = FastAPI(title="File Upload Server", description="Server for handling zip uploads only")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory if it doesn't exist
TEMP_DIR = Path(__file__).parent / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SSE endpoint to stream training logs
@app.get("/train-logs/{session_id}")
async def train_logs(session_id: str):
    def log_stream():
        print(f"Launching training: {sys.executable} cloud/ImgClass/ImgClassTrain.py {session_id}")
        try:
            # Start SageMaker training and stream logs from backend stdout
            session_dir = TEMP_DIR / session_id
            items = [item for item in session_dir.iterdir() if item.is_dir()]
            if not items:
                yield f"data: No dataset folder found after unzip\n\n"
                return
            dataset_root = str(items[0])
            aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
            aws_helper.upload_zip(dataset_root, "curate/datasets/")
            aws_helper.set_base_job_name(os.path.basename(dataset_root))
            hyperparameters = {
                'use_ai_advisor': '',
                'apply_recommendations': '',
                'save_recommendations': '',
                'epochs': 10,
                'batch_size': 32
            }
            output_path = f"s3://{aws_helper.bucket}/curate/output/"
            # Start SageMaker and capture stdout
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 "import sys; from cloud.aws import AWSHelper; aws_helper = AWSHelper('curate-sagemaker-bucket-123456789012'); aws_helper.start_sagemaker_executor(instance_type='ml.g4dn.xlarge', instance_count=1, hyperparameters={'use_ai_advisor': '', 'apply_recommendations': '', 'save_recommendations': '', 'epochs': 10, 'batch_size': 32}, output_path='s3://curate-sagemaker-bucket-123456789012/curate/output/')"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
        except Exception as e:
            print(f"Subprocess failed to start: {e}")
            yield f"data: Subprocess failed to start: {e}\n\n"
            return
        for line in process.stdout:
            yield f"data: {line}\n\n"
        print("Training subprocess finished.")
    return StreamingResponse(log_stream(), media_type="text/event-stream")

# Endpoint to extract test results after training
#@app.get("/train-results/{session_id}")
#async def train_results(session_id: str):
#    # Example: look for a results file in the session directory
#    session_dir = TEMP_DIR / session_id
#    results_path = session_dir / "test_results.json"
#    if not results_path.exists():
#        raise HTTPException(status_code=404, detail="Test results not found")
#    import json
#    with open(results_path, "r") as f:
#        results = json.load(f)
#    return results

# Train endpoint
@app.post("/train/{session_id}")
async def train(session_id: str):
    session_dir = TEMP_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    # Find the first folder (assume dataset root)
    items = [item for item in session_dir.iterdir() if item.is_dir()]
    if not items:
        raise HTTPException(status_code=400, detail="No dataset folder found after unzip")
    dataset_root = str(items[0])

    # Initialize AWS helper
    aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
    # Upload dataset to S3
    try:
        aws_helper.upload_zip(dataset_root, "curate/datasets/")
        aws_helper.set_base_job_name(os.path.basename(dataset_root))
        hyperparameters = {
            'use_ai_advisor': '',
            'apply_recommendations': '',
            'save_recommendations': '',
            'epochs': 10,
            'batch_size': 32
        }
        output_path = f"s3://{aws_helper.bucket}/curate/output/"
        aws_helper.start_sagemaker_executor(
            instance_type="ml.g4dn.xlarge",
            instance_count=1,
            hyperparameters=hyperparameters,
            output_path=output_path
        )
        return {"status": "Training Finished", "job_name": aws_helper.base_job_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Zip Upload Server is running"}

@app.post("/upload/zip")
async def upload_zip(file: UploadFile = File(...)):
    """Accept a single zip file and save it in temp_uploads."""
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted.")
    try:
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        zip_path = session_dir / file.filename
        with open(zip_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(session_dir)
        # Remove __MACOSX folder if present
        macosx_dir = session_dir / "__MACOSX"
        if macosx_dir.exists() and macosx_dir.is_dir():
            import shutil
            shutil.rmtree(macosx_dir)
        # Find the first folder (assume dataset root), ignore __MACOSX
        items = [item for item in session_dir.iterdir() if item.is_dir() and item.name != "__MACOSX"]
        if not items:
            raise HTTPException(status_code=400, detail="No dataset folder found after unzip")
        dataset_root = str(items[0])
        # Run ImgClassData and LLM task inference, save results
        try:
            img_data = ImgClassData(dataset_root)
            def count_images_in_folders(folders):
                total = 0
                per_class = {}
                for folder in folders:
                    if os.path.isdir(folder):
                        count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
                        per_class[os.path.basename(folder)] = count
                        total += count
                return total, per_class

            train_total, train_per_class = count_images_in_folders(img_data.train_folders)
            val_total, val_per_class = count_images_in_folders(img_data.val_folders)
            test_total, test_per_class = count_images_in_folders(img_data.test_folders)
            total_images = train_total + val_total + test_total

            # Get LLM-inferred task
            file_tree = img_data.json_tree
            full_prompt = f"""
            You are a data science assistant. Here is the file tree of a dataset:
            {file_tree}

            Based only on the file tree, extract the most likely ML task. 
            If the task is Image Classification return \"Image Classification\"
            If the task is Image Segmentation return \"Image Segmentation\"
            If the task is Object Detection return \"Object Detection\"
            If the task is Text Classification return \"Text Classification\"
            If the task is not any of the above return \"NONE\"
            Only return one of these exact strings. Be strict.
            """
            llm_task = "NONE"
            try:
                openai.api_key = OPENAI_API_KEY
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.1
                )
                llm_task = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM error: {e}")
                llm_task = "ERROR"

            result = {
                "session_id": session_id,
                "train_dir": img_data.train_dir,
                "val_dir": img_data.val_dir,
                "test_dir": img_data.test_dir,
                "classes": img_data.classes,
                "total_images": total_images,
                "train_images": train_total,
                "val_images": val_total,
                "test_images": test_total,
                "train_images_per_class": train_per_class,
                "val_images_per_class": val_per_class,
                "test_images_per_class": test_per_class,
                "task": llm_task
            }
            import json
            with open(session_dir / "dataset_info.json", "w") as f:
                json.dump(result, f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded zip file {file.filename}",
                "session_id": session_id,
                "file": file.filename,
                "size": len(content),
                "upload_directory": str(session_dir)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Zip upload failed: {str(e)}")

# Endpoint to get dataset info after upload and unzip
@app.get("/dataset-info/{session_id}")
async def dataset_info(session_id: str):
    session_dir = TEMP_DIR / session_id
    info_path = session_dir / "dataset_info.json"
    if not info_path.exists():
        raise HTTPException(status_code=404, detail="Dataset info not found. Please upload a zip file first.")
    import json
    with open(info_path, "r") as f:
        result = json.load(f)
    return result
