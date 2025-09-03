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
import json
import threading

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
JOB_MAP_PATH = Path(__file__).parent / "sagemaker_job_map.json"
def save_job_map(job_map):
    with open(JOB_MAP_PATH, "w") as f:
        json.dump(job_map, f)
def load_job_map():
    if JOB_MAP_PATH.exists():
        with open(JOB_MAP_PATH, "r") as f:
            return json.load(f)
    return {}

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
@app.get("/available-datasets")
async def available_datasets():
    """List available datasets in the S3 curate/datasets bucket."""
    aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
    s3_client = aws_helper.s3_client
    bucket = aws_helper.bucket
    prefix = "curate/datasets/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        datasets = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".zip"):
                datasets.append(key.replace(prefix, ""))
        return {"datasets": datasets}
    except Exception as e:
        return {"error": str(e)}
# SSE endpoint to stream training logs
@app.get("/train-logs/{session_id}")
async def train_logs(session_id: str):
    def log_stream():
        print(f"Launching training logs for session: {session_id}")
        # Try local session logs first
        session_dir = TEMP_DIR / session_id
        if session_dir.exists():
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
            process = subprocess.Popen(
                [sys.executable, "-u", "-c",
                 "import sys; from cloud.aws import AWSHelper; aws_helper = AWSHelper('curate-sagemaker-bucket-123456789012'); aws_helper.start_sagemaker_executor(instance_type='ml.g4dn.xlarge', instance_count=1, hyperparameters={'use_ai_advisor': '', 'apply_recommendations': '', 'save_recommendations': '', 'epochs': 10, 'batch_size': 32}, output_path='s3://curate-sagemaker-bucket-123456789012/curate/output/')"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in process.stdout:
                yield f"data: {line}\n\n"
            print("Training subprocess finished.")
            return
        # S3 job: stream SageMaker logs from CloudWatch
        job_map = load_job_map()
        job_name = job_map.get(session_id)
        if not job_name:
            yield f"data: No SageMaker job found for session {session_id}.\n\n"
            return
        import boto3
        import time
        logs_client = boto3.client("logs")
        log_group = "/curate/training"
        log_stream = session_id
        # Fallback to default SageMaker log group if custom logs are not found
        sm_log_group = "/aws/sagemaker/TrainingJobs"
        sm_log_stream = None  # Will be discovered via prefix = job_name
        next_token = None
        seen_events = set()
        retry_count = 0
        max_retries = 60  # Try for 3 minutes
        
        yield f"data: Looking for logs in CloudWatch log group: {log_group}, stream: {log_stream}\n\n"
        
        while retry_count < max_retries:
            try:
                # First check if log group exists
                try:
                    logs_client.describe_log_groups(logGroupNamePrefix=log_group)
                except logs_client.exceptions.ResourceNotFoundException:
                    if retry_count % 10 == 0:  # Every 30 seconds
                        yield f"data: Log group {log_group} not found yet, waiting...\n\n"
                    retry_count += 1
                    time.sleep(3)
                    # Try discovering SageMaker default logs as a fallback
                    if job_name and not sm_log_stream:
                        resp = logs_client.describe_log_streams(
                            logGroupName=sm_log_group,
                            logStreamNamePrefix=job_name
                        )
                        streams = resp.get("logStreams", [])
                        if streams:
                            sm_log_stream = streams[0]["logStreamName"]
                            next_token = None
                            yield f"data: Falling back to SageMaker logs: group {sm_log_group}, stream {sm_log_stream}\n\n"
                    continue
                
                # Check if log stream exists
                try:
                    logs_client.describe_log_streams(
                        logGroupName=log_group,
                        logStreamNamePrefix=log_stream
                    )
                except logs_client.exceptions.ResourceNotFoundException:
                    if retry_count % 10 == 0:  # Every 30 seconds
                        yield f"data: Log stream {log_stream} not found yet, waiting for training to start...\n\n"
                    # Try discovering SageMaker default logs as a fallback
                    if job_name and not sm_log_stream:
                        resp = logs_client.describe_log_streams(
                            logGroupName=sm_log_group,
                            logStreamNamePrefix=job_name
                        )
                        streams = resp.get("logStreams", [])
                        if streams:
                            sm_log_stream = streams[0]["logStreamName"]
                            next_token = None
                            yield f"data: Falling back to SageMaker logs: group {sm_log_group}, stream {sm_log_stream}\n\n"
                    retry_count += 1
                    time.sleep(3)
                    continue
                
                # Determine which log group/stream to read from
                active_group = log_group if not sm_log_stream else sm_log_group
                active_stream = log_stream if not sm_log_stream else sm_log_stream
                kwargs = {
                    "logGroupName": active_group,
                    "logStreamName": active_stream,
                    "startFromHead": True
                }
                if next_token:
                    kwargs["nextToken"] = next_token
                    
                response = logs_client.get_log_events(**kwargs)
                events = response.get("events", [])
                
                if events:
                    for event in events:
                        event_id = event["eventId"]
                        if event_id not in seen_events:
                            seen_events.add(event_id)
                            yield f"data: {event['message']}\n\n"
                    next_token = response.get("nextForwardToken")
                else:
                    if retry_count % 10 == 0:  # Every 30 seconds
                        if not sm_log_stream:
                            yield f"data: Log stream exists but no events yet, waiting...\n\n"
                        else:
                            yield f"data: SageMaker stream exists but no events yet, waiting...\n\n"
                
                retry_count += 1
                time.sleep(3)
                
            except Exception as e:
                yield f"data: Error streaming training logs: {str(e)}\n\n"
                retry_count += 1
                time.sleep(3)
                if retry_count >= max_retries:
                    break
        
        yield f"data: Stopped monitoring logs after {max_retries * 3} seconds\n\n"
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

@app.post("/train-s3/{zip_name}")
async def train_s3(zip_name: str):
    """Trigger training for a dataset zip in S3 (curate/datasets/)."""
    import uuid
    session_id = str(uuid.uuid4())
    aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
    s3_client = aws_helper.s3_client
    bucket = aws_helper.bucket
    s3_key = f"curate/datasets/{zip_name}"
    # Check if zip exists in S3
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_key)
        found = any(obj["Key"] == s3_key for obj in response.get("Contents", []))
        if not found:
            raise HTTPException(status_code=404, detail=f"Dataset {zip_name} not found in S3.")
        aws_helper.s3_path = f"s3://{bucket}/{s3_key}"
        aws_helper.set_base_job_name(zip_name.replace(".zip", ""))
        hyperparameters = {
            'use_ai_advisor': '',
            'apply_recommendations': '',
            'save_recommendations': '',
            'epochs': 10,
            'batch_size': 32,
            'zip_s3_path': aws_helper.s3_path,
            'session_id': session_id
        }
        output_path = f"s3://{bucket}/curate/output/"
        estimator = aws_helper.start_sagemaker_executor(
            instance_type="ml.g4dn.xlarge",
            instance_count=1,
            hyperparameters=hyperparameters,
            output_path=output_path,
            return_estimator=True
        )
        job_name = estimator.latest_training_job.name
        job_map = load_job_map()
        job_map[session_id] = job_name
        save_job_map(job_map)
        return {"status": "Training Started", "session_id": session_id, "job_name": job_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
