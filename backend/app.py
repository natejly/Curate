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



@app.get("/train-logs/{session_id}")
async def train_logs(session_id: str):
    import asyncio
    import os as _os
    import re
    import json
    from datetime import datetime
    import time

    def parse_epoch_metrics(log_line: str):
        """Parse epoch metrics from log lines."""
        try:
            # Look for epoch lines like: "Epoch 1: loss=0.1234, acc=0.4567, val_loss=0.7890, val_acc=0.1234"
            epoch_pattern = r'Epoch (\d+): loss=([0-9.]+), acc=([0-9.]+), val_loss=([0-9.]+), val_acc=([0-9.]+)'
            match = re.search(epoch_pattern, log_line)
            if match:
                epoch, loss, acc, val_loss, val_acc = match.groups()
                return {
                    "epoch": int(epoch),
                    "loss": float(loss),
                    "accuracy": float(acc),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Error parsing epoch metrics: {e}")
        return None

    def parse_stage_info(log_line: str):
        """Parse stage information from log lines."""
        try:
            if "STAGE 1: Training with FROZEN backbone" in log_line:
                return {"stage": 1, "type": "frozen", "message": "Stage 1: Frozen Backbone Training"}
            elif "STAGE 2: Fine-tuning with UNFROZEN top layers" in log_line:
                return {"stage": 2, "type": "fine_tuning", "message": "Stage 2: Fine-tuning"}
            elif "TRAINING JOB COMPLETED" in log_line:
                return {"stage": "completed", "type": "completed", "message": "Training Completed"}
        except Exception as e:
            print(f"Error parsing stage info: {e}")
        return None

    def parse_test_results(log_line: str):
        """Parse final test results from log lines."""
        try:
            # Look for various test result patterns
            patterns = [
                r"Test results:\s*\{([^}]+)\}",  # Test results: {dict}
                r"Test (\w+):\s*([0-9.]+)",       # Test loss: 0.1234
                r"FINAL TEST RESULTS",            # Header line
            ]

            # Check for individual metric lines first
            metric_match = re.search(r"Test (\w+):\s*([0-9.]+)", log_line)
            if metric_match:
                key, value = metric_match.groups()
                return {key: float(value)}

            # Check for full results dictionary
            dict_match = re.search(r"Test results:\s*\{([^}]+)\}", log_line)
            if dict_match:
                results_str = dict_match.group(1)
                # Parse the dictionary string
                results_dict = {}
                # Split by comma and parse key-value pairs
                pairs = [pair.strip() for pair in results_str.split(',')]
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().strip("'\"")
                        value = value.strip()
                        try:
                            # Try to convert to float
                            results_dict[key] = float(value)
                        except ValueError:
                            results_dict[key] = value.strip("'\"")
                return results_dict

            # Check for header (we'll accumulate metrics after this)
            if "FINAL TEST RESULTS" in log_line:
                return {"test_header": True}

        except Exception as e:
            print(f"Error parsing test results: {e}")
        return None

    def format_log_line(log_line: str) -> str:
        """
        Parse and format AWS CloudWatch log lines to show user-friendly output.

        Expected format: "2025-09-03T18:53:07.831000+00:00 74381aef-2535-483f-88f1-6ac6e61cc2a2 2025-09-03 18:53:07,831 - INFO - Training log saved"

        Returns: "[18:53:07] Training log saved"
        """
        try:
            # Pattern to match the AWS log format
            # Group 1: AWS timestamp (2025-09-03T18:53:07.831000+00:00)
            # Group 2: Request ID (74381aef-2535-483f-88f1-6ac6e61cc2a2)
            # Group 3: Log timestamp (2025-09-03 18:53:07,831)
            # Group 4: Log level (INFO, ERROR, WARNING, etc.)
            # Group 5: Actual message
            pattern = r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+\d{2}:\d{2})\s+[a-f0-9-]+\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+-\s+(\w+)\s+-\s+(.+)$'

            match = re.match(pattern, log_line.strip())
            if match:
                aws_timestamp, log_timestamp, log_level, message = match.groups()

                # Convert AWS timestamp to local time
                try:
                    # Parse the AWS timestamp (it's in UTC)
                    utc_time = datetime.fromisoformat(aws_timestamp.replace('Z', '+00:00'))
                    # Convert to local time
                    local_time = utc_time.astimezone()
                    # Format as HH:MM:SS
                    time_str = local_time.strftime('%H:%M:%S')
                except Exception:
                    # Fallback: use the log timestamp from the message
                    time_str = log_timestamp.split()[1].split(',')[0] if ',' in log_timestamp else log_timestamp.split()[1]

                level_emoji = {
                    'INFO': '',
                    'ERROR': '[ERROR]',
                    'WARNING': '[WARNING]',
                    'DEBUG': '[DEBUG]',
                    'CRITICAL': '[CRITICAL]'
                }.get(log_level.upper(), '📝')

                # Clean up the message (remove extra whitespace)
                clean_message = message.strip()

                # Format the final output
                return f"[{time_str}] {level_emoji} {clean_message}"

        except Exception as e:
            # If parsing fails, return the original line
            print(f"[DEBUG] Failed to parse log line: {log_line} - Error: {e}")
            return log_line.strip()

    region = _os.environ.get('AWS_REGION') or _os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1'

    async def log_stream():
        print(f"[DEBUG] Starting integrated log and metrics stream for session: {session_id} in {region}")

        # Initialize metrics tracking
        metrics_data = {
            "session_id": session_id,
            "stage1_metrics": [],
            "stage2_metrics": [],
            "current_stage": 1,
            "training_status": "initializing",
            "stage_info": None,
            "final_test_results": None
        }

        # Keep waiting and retrying indefinitely until logs appear or training finishes
        retry_count = 0
        training_finished = False

        while not training_finished:
            try:
                # Try with specific log stream name first
                process = await asyncio.create_subprocess_exec(
                    "aws", "logs", "tail", "/curate/training",
                    "--region", region,
                    "--follow",
                    "--log-stream-names", session_id,
                    "--no-paginate",  # Add this flag
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**_os.environ, "AWS_PAGER": "", "PYTHONUNBUFFERED": "1"}  # Force unbuffered
                )

                # Wait a bit to see if process starts successfully
                await asyncio.sleep(2)

                if process.returncode is not None:
                    # Process failed, wait and retry
                    retry_count += 1
                    await asyncio.sleep(5)  # Wait 5 seconds before retry
                    continue

                # Process is running, start streaming logs and metrics
                last_heartbeat = asyncio.get_event_loop().time()
                heartbeat_interval = 60  # Send heartbeat every 60 seconds for long training
                last_metrics_update = asyncio.get_event_loop().time()
                metrics_update_interval = 2  # Send metrics every 2 seconds

                while True:
                    # Check if we need to send a heartbeat
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        yield f"data: {json.dumps({'type': 'log', 'message': f'[HEARTBEAT] Training in progress for {session_id}...'})}\n\n"
                        last_heartbeat = current_time

                    # Send metrics update if needed
                    if current_time - last_metrics_update > metrics_update_interval:
                        yield f"data: {json.dumps({'type': 'metrics', 'data': metrics_data})}\n\n"
                        last_metrics_update = current_time

                    try:
                        line = await process.stdout.readline()
                    except Exception as read_error:
                        print(f"[ERROR] Error reading line: {read_error}")
                        await asyncio.sleep(1)
                        continue

                    if not line:
                        # Check if the process has terminated
                        if process.returncode is not None:
                            yield f"data: {json.dumps({'type': 'log', 'message': f'Log stream ended (process exit code: {process.returncode})'})}\n\n"
                            # If process ended but we haven't seen "TRAINING FINISHED", retry
                            if not training_finished:
                                print(f"[DEBUG] Process ended but training not finished, will retry...")
                                await asyncio.sleep(5)
                                break  # Break inner loop to retry
                            else:
                                return  # Training finished, end stream
                        # Process is still running but no data, wait a bit
                        await asyncio.sleep(2)
                        continue

                    decoded_line = line.decode('utf-8', errors='replace').rstrip()
                    if decoded_line.strip():  # Only send non-empty lines
                        # Debug: Check for test-related lines
                        if "Test" in decoded_line or "test" in decoded_line.lower():
                            print(f"[DEBUG] Found test-related line: {decoded_line}")

                        # Parse and format the log line
                        formatted_line = format_log_line(decoded_line)

                        if formatted_line:  # Only send if we successfully parsed it
                            # Parse metrics from the log line
                            epoch_data = parse_epoch_metrics(decoded_line)
                            if epoch_data:
                                if metrics_data["current_stage"] == 1:
                                    metrics_data["stage1_metrics"].append(epoch_data)
                                else:
                                    metrics_data["stage2_metrics"].append(epoch_data)

                            # Parse test results
                            test_results = parse_test_results(decoded_line)
                            if test_results:
                                if test_results.get("test_header"):
                                    # Initialize test results collection
                                    metrics_data["final_test_results"] = {}
                                    print(f"[DEBUG] Started collecting test results")
                                elif len(test_results) == 1 and not test_results.get("test_header"):
                                    # Single metric - accumulate it
                                    if not metrics_data.get("final_test_results"):
                                        metrics_data["final_test_results"] = {}
                                    metrics_data["final_test_results"].update(test_results)
                                    print(f"[DEBUG] Added test metric: {test_results}")
                                else:
                                    # Full results dictionary
                                    metrics_data["final_test_results"] = test_results
                                    print(f"[DEBUG] Parsed full test results: {test_results}")

                            # Parse stage information
                            stage_data = parse_stage_info(decoded_line)
                            if stage_data:
                                if stage_data["stage"] == 2:
                                    metrics_data["current_stage"] = 2
                                elif stage_data["stage"] == "completed":
                                    metrics_data["training_status"] = "completed"
                                metrics_data["stage_info"] = stage_data

                            # Check for training completion
                            if "TRAINING JOB COMPLETED" in formatted_line or "Training completed" in formatted_line:
                                training_finished = True
                                yield f"data: {json.dumps({'type': 'log', 'message': f'🎉 {formatted_line}'})}\n\n"
                                yield f"data: {json.dumps({'type': 'log', 'message': f'Training completed for session {session_id}'})}\n\n"
                                # Send final metrics
                                yield f"data: {json.dumps({'type': 'metrics', 'data': metrics_data})}\n\n"
                                return  # End the stream

                            # Send the log line
                            yield f"data: {json.dumps({'type': 'log', 'message': formatted_line})}\n\n"

                # If we get here from inner loop break, continue outer loop to retry

            except Exception as e:
                retry_count += 1
                print(f"[ERROR] Log streaming error: {e}")
                yield f"data: {json.dumps({'type': 'log', 'message': f'Connection error (attempt {retry_count}): {str(e)}'})}\n\n"
                yield f"data: {json.dumps({'type': 'log', 'message': 'Retrying in 10 seconds...'})}\n\n"
                await asyncio.sleep(10)
            finally:
                try:
                    if 'process' in locals() and process and process.returncode is None:
                        process.terminate()
                        await asyncio.sleep(0.5)  # Give it time to terminate gracefully
                        if process.returncode is None:
                            process.kill()  # Force kill if it didn't terminate
                        await process.wait()
                except Exception as cleanup_error:
                    print(f"[ERROR] Cleanup error: {cleanup_error}")

        # This should never be reached due to the return statements above
        yield f"data: {json.dumps({'type': 'log', 'message': f'Training stream ended for session {session_id}'})}\n\n"
    
    # Return StreamingResponse with proper SSE headers
    return StreamingResponse(
        log_stream(), 
        media_type="text/plain",  # Changed from text/event-stream
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

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
            'batch_size': 32,
            'session_id': session_id
        }
        output_path = f"s3://{aws_helper.bucket}/curate/output/"
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

from typing import Optional
from fastapi import Request

# Support both styles: session_id in path or provided in JSON body
@app.post("/train-s3/{zip_name}")
@app.post("/train-s3/{zip_name}/{session_id}")
async def train_s3(zip_name: str, request: Request, session_id: Optional[str] = None):
    """Trigger training for a dataset zip in S3 (curate/datasets/) with a single client-provided session_id."""
    # Prefer session_id from path parameter first, then from JSON body
    if session_id is None:
        try:
            body = await request.json()
            session_id = body.get("session_id") if isinstance(body, dict) else None
        except Exception:
            session_id = None
    
    # Only generate a new session_id if none was provided
    if not session_id:
        from uuid import uuid4
        session_id = str(uuid4())
        print(f"[DEBUG] Generated new session_id: {session_id}")
    else:
        print(f"[DEBUG] Using provided session_id: {session_id}")
    
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

