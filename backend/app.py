from fastapi.responses import StreamingResponse, FileResponse
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



@app.get("/test-s3")
async def test_s3():
    """Test S3 connectivity."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client('s3')
        bucket_name = "curate-sagemaker-bucket-123456789012"

        # Try to list objects
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=5)

        if 'Contents' in response:
            objects = [obj['Key'] for obj in response['Contents']]
            return {"s3_status": "connected", "bucket": bucket_name, "sample_objects": objects[:3]}
        else:
            return {"s3_status": "connected", "bucket": bucket_name, "message": "Bucket exists but no objects found"}

    except ClientError as e:
        return {"s3_status": "error", "error": str(e), "error_code": e.response['Error']['Code']}
    except Exception as e:
        return {"s3_status": "error", "error": str(e)}

@app.get("/debug-training-log/{session_id}")
async def debug_training_log(session_id: str):
    """Debug endpoint to view raw training log content from local files or S3."""
    import os
    from pathlib import Path
    import boto3
    from botocore.exceptions import ClientError
    try:
            print("looking")
            # Initialize S3 client
            s3_client = boto3.client('s3')
            bucket_name = "curate-sagemaker-bucket-123456789012"

            # Try to get the training log from S3
            training_log_key = f"curate/logs/{session_id}/training_log.json"

            response = s3_client.get_object(Bucket=bucket_name, Key=training_log_key)
            log_content = response['Body'].read().decode('utf-8')
            print(f"[DEBUG] Found training log in S3 for session {session_id}")
            return {"log_content": log_content}


    except Exception as e:
        return {"error": f"Failed to debug training log: {str(e)}"}

@app.get("/debug-test-metrics/{session_id}")
async def debug_test_metrics(session_id: str):
    """Debug endpoint to analyze test metrics parsing and display."""
    try:
        # Get model stats which includes parsed test metrics
        stats_response = await get_model_stats(session_id)
        
        # Get raw training log for comparison
        debug_log_response = await debug_training_log(session_id)
        
        # Try to extract test metrics from raw log
        raw_test_metrics = {}
        if "log_content" in debug_log_response:
            try:
                log_data = json.loads(debug_log_response["log_content"])
                
                # Extract test metrics from each iteration
                for iteration_key, iteration_data in log_data.items():
                    if isinstance(iteration_data, dict):
                        iteration_info = {
                            "iteration_key": iteration_key,
                            "training_type": iteration_data.get("training_type"),
                            "optimization_iteration": iteration_data.get("optimization_iteration"),
                            "test_metrics": None,
                            "test_data": None
                        }
                        
                        # Check for test metrics in different locations
                        if "test_metrics" in iteration_data:
                            iteration_info["test_metrics"] = iteration_data["test_metrics"]
                        elif "test" in iteration_data:
                            iteration_info["test_data"] = iteration_data["test"]
                        
                        if iteration_info["test_metrics"] or iteration_info["test_data"]:
                            raw_test_metrics[iteration_key] = iteration_info
            except Exception as parse_error:
                raw_test_metrics["parse_error"] = str(parse_error)
        
        return {
            "session_id": session_id,
            "parsed_stats": stats_response,
            "raw_test_metrics": raw_test_metrics,
            "debug_info": {
                "stats_available": "error" not in stats_response,
                "raw_log_available": "log_content" in debug_log_response,
                "test_metrics_found": len([k for k in raw_test_metrics.keys() if k != "parse_error"]),
                "parsing_tips": [
                    "Check if test metrics are stored in 'test' or 'test_metrics' keys",
                    "Verify metric names match expected format (accuracy, loss, etc.)",
                    "Ensure numeric values are properly formatted",
                    "Look for optimization_iteration markers"
                ]
            }
        }
        
    except Exception as e:
        return {"error": f"Failed to debug test metrics: {str(e)}", "session_id": session_id}

@app.get("/model-stats/{session_id}")
async def get_model_stats(session_id: str):
    """Get training statistics for a specific model from local training log or S3."""
    import os
    from pathlib import Path
    import boto3
    from botocore.exceptions import ClientError

    try:
        # Get the local training log path
        backend_dir = Path(__file__).parent
        curate_dir = backend_dir.parent / "curate"

        # Try multiple possible locations for the training log
        possible_paths = [
            curate_dir / "logs" / session_id / "training_log.json",
            curate_dir / "models" / session_id / "training_log.json"
        ]

        training_log_path = None
        for path in possible_paths:
            if path.exists():
                training_log_path = path
                break

        # If not found locally, try S3
        if not training_log_path:
            try:
                # Initialize S3 client
                s3_client = boto3.client('s3')
                bucket_name = "curate-sagemaker-bucket-123456789012"

                # Try to get the training log from S3
                training_log_key = f"curate/logs/{session_id}/training_log.json"

                response = s3_client.get_object(Bucket=bucket_name, Key=training_log_key)
                log_content = response['Body'].read().decode('utf-8')
                log_data = json.loads(log_content)
                print(f"[DEBUG] Found training log in S3 for session {session_id}")

            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return {"error": f"Training log not found for session {session_id}. Searched local: {[str(p) for p in possible_paths]}, S3 key: {training_log_key}"}
                else:
                    return {"error": f"S3 error: {e.response['Error']['Message']}"}
            except json.JSONDecodeError:
                return {"error": "Invalid training log format from S3"}
        else:
            # Read and parse the local training log
            try:
                with open(training_log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    log_data = json.loads(log_content)
                print(f"[DEBUG] Found training log locally for session {session_id}: {training_log_path}")
            except json.JSONDecodeError as e:
                return {"error": f"Invalid training log format: {e}"}
            except Exception as e:
                return {"error": f"Failed to read training log: {e}"}

        # Parse the training log to extract final stats
        stats = parse_training_stats(log_data)

        return {
            "session_id": session_id,
            "stats": stats
        }

    except Exception as e:
        return {"error": f"Failed to get model stats: {str(e)}"}

def parse_training_stats(log_data):
    """Parse training log data to extract final statistics."""
    try:
        stats = {
            # Training metrics (from epochs)
            "final_training_accuracy": None,
            "final_training_loss": None,
            "final_val_accuracy": None,
            "final_val_loss": None,
            "total_epochs": 0,
            "best_epoch": None,
            
            # Test metrics (from model.evaluate())
            "test_accuracy": None,
            "test_loss": None,
            
            # General info
            "training_time": None,
            "dataset_name": None,
            "dataset_path": None,
            "img_size": None,
            "num_classes": None,
            "base_model_name": None,
            
            # Model details
            "model_total_parameters": None,
            "model_trainable_parameters": None,
            "model_non_trainable_parameters": None,
            "base_model_layers": None,
            "base_model_trainable": None,
        }

        # Try to extract from the log data structure
        if isinstance(log_data, dict):
            # Handle the actual TrainingLog structure
            # The log contains iterations with different formats
            latest_iteration = None
            latest_timestamp = None

            for iteration_key, iteration_data in log_data.items():
                if isinstance(iteration_data, dict) and 'timestamp' in iteration_data:
                    if latest_timestamp is None or iteration_data['timestamp'] > latest_timestamp:
                        latest_timestamp = iteration_data['timestamp']
                        latest_iteration = iteration_data

            if latest_iteration:
                # Extract parameters
                if 'params' in latest_iteration:
                    params = latest_iteration['params']
                    stats['dataset_name'] = params.get('dataset_name')
                    stats['dataset_path'] = params.get('dataset_path')
                    stats['img_size'] = params.get('img_size')
                    stats['num_classes'] = params.get('num_classes')
                    stats['base_model_name'] = params.get('base_model_name')
                    
                    # Extract model details
                    stats['model_total_parameters'] = params.get('model_total_parameters')
                    stats['model_trainable_parameters'] = params.get('model_trainable_parameters')
                    stats['model_non_trainable_parameters'] = params.get('model_non_trainable_parameters')
                    stats['base_model_layers'] = params.get('base_model_layers')
                    stats['base_model_trainable'] = params.get('base_model_trainable')

                # Extract training history based on training type
                training_type = latest_iteration.get('training_type', 'single_stage')

                if training_type == 'two_stage':
                    # Two-stage training
                    if 'stage2_logs' in latest_iteration and latest_iteration['stage2_logs']:
                        history = latest_iteration['stage2_logs']
                    elif 'stage1_logs' in latest_iteration and latest_iteration['stage1_logs']:
                        history = latest_iteration['stage1_logs']
                    else:
                        history = None
                else:
                    # Single-stage training
                    history = latest_iteration.get('logs')

                if history:
                    # Get final TRAINING metrics (from epochs)
                    if 'accuracy' in history and isinstance(history['accuracy'], list) and history['accuracy']:
                        stats['final_training_accuracy'] = history['accuracy'][-1]
                    if 'loss' in history and isinstance(history['loss'], list) and history['loss']:
                        stats['final_training_loss'] = history['loss'][-1]
                    if 'val_accuracy' in history and isinstance(history['val_accuracy'], list) and history['val_accuracy']:
                        stats['final_val_accuracy'] = history['val_accuracy'][-1]
                    if 'val_loss' in history and isinstance(history['val_loss'], list) and history['val_loss']:
                        stats['final_val_loss'] = history['val_loss'][-1]

                    # Get total epochs
                    if 'accuracy' in history and isinstance(history['accuracy'], list):
                        stats['total_epochs'] = len(history['accuracy'])

                    # Find best epoch (highest validation accuracy)
                    if 'val_accuracy' in history and isinstance(history['val_accuracy'], list) and history['val_accuracy']:
                        best_val_acc = max(history['val_accuracy'])
                        stats['best_epoch'] = history['val_accuracy'].index(best_val_acc) + 1

                # Extract TEST metrics if available (from model.evaluate())
                if training_type == 'two_stage' and 'test_metrics' in latest_iteration:
                    test_metrics = latest_iteration['test_metrics']
                    if test_metrics:
                        # Store test metrics separately from training metrics
                        if 'accuracy' in test_metrics:
                            stats['test_accuracy'] = test_metrics['accuracy']
                        if 'loss' in test_metrics:
                            stats['test_loss'] = test_metrics['loss']
                        # Note: test metrics shouldn't have val_accuracy/val_loss as those are training concepts
                        
                elif training_type == 'single_stage' and 'test' in latest_iteration:
                    test_metrics = latest_iteration['test']
                    if test_metrics:
                        # Store test metrics separately from training metrics
                        if 'accuracy' in test_metrics:
                            stats['test_accuracy'] = test_metrics['accuracy']
                        if 'loss' in test_metrics:
                            stats['test_loss'] = test_metrics['loss']
                        # Note: test metrics shouldn't have val_accuracy/val_loss as those are training concepts

        return stats

    except Exception as e:
        print(f"Error parsing training stats: {e}")
        import traceback
        traceback.print_exc()
        return {
            "final_accuracy": None,
            "final_loss": None,
            "final_val_accuracy": None,
            "final_val_loss": None,
            "total_epochs": 0,
            "best_epoch": None,
            "training_time": None,
            "dataset_name": None,
            "img_size": None,
            "num_classes": None,
            "base_model_name": None,
        }

@app.get("/list-models")
async def list_models():
    """List all trained ONNX models from S3 bucket, similar to available datasets."""
    import boto3
    from botocore.exceptions import ClientError

    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        bucket_name = "curate-sagemaker-bucket-123456789012"

        # List all objects in the models directory
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix="curate/models/"
        )

        if 'Contents' not in response:
            return {"models": []}

        # Find ONNX files for each session
        models = []
        session_onnx_files = {}  # Track latest ONNX file per session

        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('/') or not key.startswith('curate/models/'):
                continue

            # Parse session ID and model name from key
            # Key format: curate/models/{session_id}/{model_name}
            parts = key.split('/')
            if len(parts) >= 4:  # curate/models/session_id/model_name
                session_id = parts[2]
                model_name = '/'.join(parts[3:])

                # Only include ONNX files (skip training logs and other files)
                if model_name.endswith('.onnx') and model_name != "training_log.json" and not model_name.startswith("__"):
                    # Keep track of the latest ONNX file for each session
                    if session_id not in session_onnx_files or obj['LastModified'] > session_onnx_files[session_id]['LastModified']:
                        session_onnx_files[session_id] = {
                            "session_id": session_id,
                            "filename": model_name,
                            "s3_key": key,
                            "size": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "url": f"https://{bucket_name}.s3.amazonaws.com/{key}"
                        }

        # Convert to list and sort by last modified (newest first)
        models = list(session_onnx_files.values())
        models.sort(key=lambda x: x['last_modified'], reverse=True)

        return {"models": models}

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            return {"error": f"S3 bucket '{bucket_name}' not found. Please check your S3 configuration."}
        elif e.response['Error']['Code'] == 'AccessDenied':
            return {"error": "Access denied to S3 bucket. Please check your AWS credentials and permissions."}
        else:
            return {"error": f"S3 error: {e.response['Error']['Message']}"}
    except Exception as e:
        return {"error": f"Failed to list models: {str(e)}"}

@app.post("/upload-models/{session_id}")
async def upload_models_endpoint(session_id: str):
    """Manually upload models for a session to S3."""
    import subprocess
    import sys
    import os

    try:
        # Get the path to s3_uploader.py
        uploader_path = os.path.join(os.path.dirname(__file__), 's3_uploader.py')

        if not os.path.exists(uploader_path):
            return {"error": f"S3 uploader script not found at: {uploader_path}"}

        print(f"Uploading models for session {session_id} to S3...")

        # Run the upload script
        result = subprocess.run([
            sys.executable, uploader_path, session_id, "--models"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        if result.returncode == 0:
            return {
                "success": True,
                "message": f"Models for session {session_id} uploaded to S3 successfully",
                "output": result.stdout.strip()
            }
        else:
            return {
                "success": False,
                "error": f"Upload failed with exit code {result.returncode}",
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }

    except subprocess.TimeoutExpired:
        return {"error": "Upload timed out after 5 minutes"}
    except Exception as e:
        return {"error": f"Failed to upload models: {str(e)}"}

@app.get("/download-model/{session_id}")
async def download_model(session_id: str, filename: str = None):
    """Download trained model from S3."""
    import boto3
    from botocore.exceptions import ClientError

    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        bucket_name = "curate-sagemaker-bucket-123456789012"

        # If no filename provided, try to find the ONNX file for this session
        if not filename:
            # List objects in the session directory to find the ONNX file
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=f"curate/models/{session_id}/"
            )

            if 'Contents' not in response:
                return {"error": f"No models found for session {session_id}"}

            # Find the ONNX file
            onnx_file = None
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.onnx') and not key.endswith('/') and 'training_log' not in key:
                    onnx_file = key
                    filename = key.split('/')[-1]  # Extract filename from path
                    break

            if not onnx_file:
                return {"error": f"No ONNX model found for session {session_id}"}

            model_key = onnx_file
        else:
            # Use the provided filename
            model_key = f"curate/models/{session_id}/{filename}"

        # Check if the file exists in S3
        try:
            s3_client.head_object(Bucket=bucket_name, Key=model_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return {"error": f"Model not found: {model_key}"}
            else:
                return {"error": f"S3 error: {str(e)}"}

        # Generate presigned URL for download (expires in 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': model_key,
                'ResponseContentDisposition': f'attachment; filename="{filename}"'
            },
            ExpiresIn=3600  # 1 hour
        )

        return {"download_url": presigned_url, "filename": filename}

    except Exception as e:
        return {"error": f"Failed to generate download URL: {str(e)}"}

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
        """Parse final test results from log lines - ONLY from model.evaluate() calls, not training epochs."""
        try:
            # STRICT PARSING: Only parse test results from actual model.evaluate() calls
            
            # Pattern 1: Look for evaluation section start marker
            if "=== EVALUATE: Test set ===" in log_line:
                print(f"[DEBUG] Found test set evaluation start marker")
                return {"evaluation_start": True}
            
            # Pattern 2: EXACT match for the evaluate() output dictionary
            # Format: Test results: {'loss': 0.1234, 'accuracy': 0.8765}
            # This is from ImgClassTrain.py line 336: print(f"Test results: {metrics}")
            # MUST start with exactly "Test results:" - not just contain it
            if log_line.strip().startswith("Test results:") and "{" in log_line and "}" in log_line:
                dict_match = re.search(r"Test results:\s*\{([^}]+)\}", log_line)
                if dict_match:
                    results_str = dict_match.group(1)
                    print(f"[DEBUG] Found EXACT 'Test results:' dictionary: {results_str}")
                    
                    # Parse the dictionary string - handle both quoted and unquoted keys
                    results_dict = {}
                    # Match patterns like 'loss': 0.1234 or loss: 0.1234
                    pairs = re.findall(r"['\"]?(\w+)['\"]?\s*:\s*([0-9.eE-]+)", results_str)
                    for key, value in pairs:
                        try:
                            results_dict[key] = float(value)
                        except ValueError:
                            results_dict[key] = value
                    
                    print(f"[DEBUG] Parsed ACTUAL test results from model.evaluate(): {results_dict}")
                    return results_dict

            # Pattern 3: FINAL TEST RESULTS header from logging
            # This is from ImgClassTrain.py line 342: logger.info(f"=== FINAL TEST RESULTS ===")
            if "=== FINAL TEST RESULTS ===" in log_line:
                print(f"[DEBUG] Found FINAL TEST RESULTS logging header")
                return {"test_header": True}
            
            # Pattern 4: Individual test metric lines from logging (VERY STRICT)
            # Format: Test loss: 0.1234 or Test accuracy: 0.8765
            # This is from ImgClassTrain.py line 344: logger.info(f"Test {key}: {value}")
            # EXCLUDE anything that looks like epoch training (contains "Epoch", "epoch", "=", or comma-separated values)
            if not any(word in log_line for word in ["Epoch", "epoch", "=", ","]):
                # Very strict pattern - must start with "Test " followed by metric name and colon
                metric_match = re.search(r"^.*Test\s+(\w+):\s*([0-9.eE-]+).*$", log_line)
                if metric_match:
                    key, value = metric_match.groups()
                    result = {key: float(value)}
                    print(f"[DEBUG] Found individual test metric from logging: {result}")
                    return result

            # Pattern 5: Test accuracy for early stopping evaluation line
            # This is from ImgClassTrain.py line 348: logger.info(f"Test accuracy for early stopping evaluation: {metrics['accuracy']:.4f}")
            early_stop_match = re.search(r"Test accuracy for early stopping evaluation:\s*([0-9.eE-]+)", log_line)
            if early_stop_match:
                accuracy = float(early_stop_match.group(1))
                result = {"accuracy": accuracy}
                print(f"[DEBUG] Found early stopping test accuracy: {result}")
                return result

        except Exception as e:
            print(f"[ERROR] Error parsing test results from line '{log_line.strip()}': {e}")
            import traceback
            traceback.print_exc()
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

    def extract_optimization_iterations(session_id: str):
        """Extract optimization iterations from training log if available."""
        import os
        from pathlib import Path
        import boto3
        from botocore.exceptions import ClientError
        
        try:
            # Try to find training log locally first
            backend_dir = Path(__file__).parent
            curate_dir = backend_dir.parent
            possible_paths = [
                curate_dir / "logs" / session_id / "training_log.json",
                curate_dir / "models" / session_id / "training_log.json",
                Path("/opt/ml/model") / "training_log.json"  # SageMaker model directory
            ]
            
            training_log_data = None
            
            # Check local paths
            for path in possible_paths:
                if path.exists():
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            training_log_data = json.loads(f.read())
                        print(f"[DEBUG] Found training log locally: {path}")
                        break
                    except Exception as e:
                        print(f"[DEBUG] Failed to read local training log {path}: {e}")
                        continue
            
            # If not found locally, try S3
            if not training_log_data:
                try:
                    s3_client = boto3.client('s3')
                    bucket_name = "curate-sagemaker-bucket-123456789012"
                    training_log_key = f"curate/logs/{session_id}/training_log.json"
                    
                    response = s3_client.get_object(Bucket=bucket_name, Key=training_log_key)
                    log_content = response['Body'].read().decode('utf-8')
                    training_log_data = json.loads(log_content)
                    print(f"[DEBUG] Found training log in S3: {training_log_key}")
                except ClientError as e:
                    if e.response['Error']['Code'] != 'NoSuchKey':
                        print(f"[DEBUG] S3 error accessing training log: {e}")
                except Exception as e:
                    print(f"[DEBUG] Error accessing S3 training log: {e}")
            
            # Extract optimization iterations from training log
            optimization_iterations = []
            if training_log_data and isinstance(training_log_data, dict):
                for iteration_key, iteration_data in training_log_data.items():
                    if (isinstance(iteration_data, dict) and 
                        iteration_data.get('is_optimization') and 
                        iteration_data.get('optimization_iteration')):
                        
                        # Extract test results
                        test_results = {}
                        if 'test' in iteration_data:
                            test_results = iteration_data['test']
                        elif 'test_metrics' in iteration_data:
                            test_results = iteration_data['test_metrics']
                        
                        # Create optimization iteration entry
                        opt_iteration = {
                            'iteration': iteration_data['optimization_iteration'],
                            'timestamp': iteration_data.get('timestamp', ''),
                            'test_results': test_results,
                            'training_type': iteration_data.get('training_type', 'unknown'),
                            'is_optimization': True
                        }
                        
                        # Add AI recommendations if available
                        if 'ai_advisor' in iteration_data:
                            opt_iteration['ai_recommendations'] = iteration_data['ai_advisor']
                        
                        optimization_iterations.append(opt_iteration)
                
                # Sort by iteration number
                optimization_iterations.sort(key=lambda x: x['iteration'])
                print(f"[DEBUG] Extracted {len(optimization_iterations)} optimization iterations")
            
            return optimization_iterations
            
        except Exception as e:
            print(f"[DEBUG] Error extracting optimization iterations: {e}")
            return []

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
            "final_test_results": None,
            "optimization_iterations": [],
            "current_optimization_iteration": 0,  # Track current optimization iteration
            "in_evaluation_phase": False  # Track when we're parsing actual test evaluation results
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
                last_optimization_check = asyncio.get_event_loop().time()
                optimization_check_interval = 10  # Check for optimization iterations every 10 seconds

                while True:
                    # Check if we need to send a heartbeat
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat > heartbeat_interval:
                        yield f"data: {json.dumps({'type': 'log', 'message': f'[HEARTBEAT] Training in progress for {session_id}...'})}\n\n"
                        last_heartbeat = current_time

                    # Check for optimization iterations periodically
                    if current_time - last_optimization_check > optimization_check_interval:
                        try:
                            optimization_iterations = extract_optimization_iterations(session_id)
                            if optimization_iterations and len(optimization_iterations) != len(metrics_data["optimization_iterations"]):
                                metrics_data["optimization_iterations"] = optimization_iterations
                                print(f"[DEBUG] Updated optimization iterations: {len(optimization_iterations)} found")
                                # Send updated metrics
                                yield f"data: {json.dumps({'type': 'metrics', 'data': metrics_data})}\n\n"
                        except Exception as e:
                            print(f"[DEBUG] Error checking optimization iterations: {e}")
                        last_optimization_check = current_time

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
                            # Check for optimization iteration start to reset epoch metrics
                            if "=== OPTIMIZATION ITERATION" in decoded_line:
                                # Extract iteration number
                                import re
                                iteration_match = re.search(r'OPTIMIZATION ITERATION (\d+)', decoded_line)
                                if iteration_match:
                                    new_iteration = int(iteration_match.group(1))
                                    if new_iteration > metrics_data["current_optimization_iteration"]:
                                        print(f"[DEBUG] Starting optimization iteration {new_iteration}, resetting epoch metrics")
                                        metrics_data["current_optimization_iteration"] = new_iteration
                                        # Reset epoch metrics for new iteration
                                        metrics_data["stage1_metrics"] = []
                                        metrics_data["stage2_metrics"] = []
                                        metrics_data["current_stage"] = 1
                            
                            # Parse metrics from the log line
                            epoch_data = parse_epoch_metrics(decoded_line)
                            if epoch_data:
                                if metrics_data["current_stage"] == 1:
                                    metrics_data["stage1_metrics"].append(epoch_data)
                                else:
                                    metrics_data["stage2_metrics"].append(epoch_data)
                                
                                # Send metrics update every few epochs to keep frontend updated
                                total_epochs = len(metrics_data["stage1_metrics"]) + len(metrics_data["stage2_metrics"])
                                if total_epochs % 3 == 0:  # Send every 3 epochs to avoid overwhelming
                                    yield f"data: {json.dumps({'type': 'metrics', 'data': metrics_data})}\n\n"

                            # Parse test results
                            test_results = parse_test_results(decoded_line)
                            if test_results:
                                # Handle evaluation phase markers
                                if test_results.get("evaluation_start"):
                                    metrics_data["in_evaluation_phase"] = True
                                    print(f"[DEBUG] Entered test evaluation phase")
                                elif test_results.get("test_header"):
                                    # Initialize test results collection and mark evaluation phase
                                    metrics_data["in_evaluation_phase"] = True
                                    metrics_data["final_test_results"] = {}
                                    print(f"[DEBUG] Started collecting FINAL test results (from model.evaluate())")
                                elif len(test_results) == 1 and not any(k in test_results for k in ["test_header", "evaluation_start"]):
                                    # Single metric from evaluation phase - accumulate it
                                    if metrics_data.get("in_evaluation_phase"):
                                        if not metrics_data.get("final_test_results"):
                                            metrics_data["final_test_results"] = {}
                                        metrics_data["final_test_results"].update(test_results)
                                        print(f"[DEBUG] Added test metric from evaluation: {test_results}")
                                    else:
                                        print(f"[DEBUG] Ignoring test metric outside evaluation phase: {test_results}")
                                else:
                                    # Full results dictionary from model.evaluate()
                                    metrics_data["final_test_results"] = test_results
                                    metrics_data["in_evaluation_phase"] = False  # Reset after getting full results
                                    print(f"[DEBUG] Parsed complete test results from model.evaluate(): {test_results}")

                                # Check if we have complete test results and are in an optimization iteration
                                if (metrics_data["current_optimization_iteration"] > 0 and 
                                    metrics_data.get("final_test_results") and 
                                    len(metrics_data["final_test_results"]) >= 2 and
                                    "loss" in metrics_data["final_test_results"] and 
                                    "accuracy" in metrics_data["final_test_results"]):
                                    
                                    # Create optimization iteration entry
                                    optimization_iteration = {
                                        "iteration": metrics_data["current_optimization_iteration"],
                                        "timestamp": datetime.now().isoformat(),
                                        "test_results": metrics_data["final_test_results"].copy(),
                                        "training_type": "optimization",
                                        "is_optimization": True
                                    }
                                    
                                    # Add to optimization iterations list
                                    if "optimization_iterations" not in metrics_data:
                                        metrics_data["optimization_iterations"] = []
                                    
                                    # Check if this iteration already exists (avoid duplicates)
                                    existing_iteration = next((iter for iter in metrics_data["optimization_iterations"] 
                                                             if iter["iteration"] == metrics_data["current_optimization_iteration"]), None)
                                    
                                    if not existing_iteration:
                                        metrics_data["optimization_iterations"].append(optimization_iteration)
                                        print(f"[DEBUG] Added optimization iteration {metrics_data['current_optimization_iteration']} with results: {metrics_data['final_test_results']}")
                                    else:
                                        # Update existing iteration with new results
                                        existing_iteration["test_results"] = metrics_data["final_test_results"].copy()
                                        existing_iteration["timestamp"] = datetime.now().isoformat()
                                        print(f"[DEBUG] Updated optimization iteration {metrics_data['current_optimization_iteration']} with results: {metrics_data['final_test_results']}")
                                    
                                    # Sort iterations by iteration number
                                    metrics_data["optimization_iterations"].sort(key=lambda x: x["iteration"])
                                    
                                    # Send updated metrics to frontend immediately
                                    yield f"data: {json.dumps({'type': 'metrics', 'data': metrics_data})}\n\n"
                                    print(f"[DEBUG] Sent optimization iteration {metrics_data['current_optimization_iteration']} to frontend")

                            # Parse stage information
                            stage_data = parse_stage_info(decoded_line)
                            if stage_data:
                                if stage_data["stage"] == 2:
                                    metrics_data["current_stage"] = 2
                                elif stage_data["stage"] == "completed":
                                    metrics_data["training_status"] = "completed"
                                metrics_data["stage_info"] = stage_data
                                # Send metrics update when stage changes
                                yield f"data: {json.dumps({'type': 'metrics', 'data': metrics_data})}\n\n"

                            # Check for training completion
                            if "TRAINING JOB COMPLETED" in formatted_line or "Training completed" in formatted_line:
                                training_finished = True
                                yield f"data: {json.dumps({'type': 'log', 'message': f'🎉 {formatted_line}'})}\n\n"
                                yield f"data: {json.dumps({'type': 'log', 'message': f'Training completed for session {session_id}'})}\n\n"
                                # Get final optimization iterations before sending final metrics
                                try:
                                    final_optimization_iterations = extract_optimization_iterations(session_id)
                                    if final_optimization_iterations:
                                        metrics_data["optimization_iterations"] = final_optimization_iterations
                                        print(f"[DEBUG] Final optimization iterations: {len(final_optimization_iterations)} found")
                                except Exception as e:
                                    print(f"[DEBUG] Error getting final optimization iterations: {e}")
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
            'epochs': 10,
            'batch_size': 32,
            'session_id': session_id
        }
        output_path = f"s3://{aws_helper.bucket}/curate/output/"
        estimator = aws_helper.start_sagemaker_executor(
            instance_type="ml.g5.4xlarge",
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
    """Accept a single zip file and save it in temp_uploads. Processing happens asynchronously."""
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted.")
    try:
        session_id = str(uuid.uuid4())
        session_dir = TEMP_DIR / session_id
        session_dir.mkdir(exist_ok=True)

        # Save the uploaded zip file
        zip_path = session_dir / file.filename
        with open(zip_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Create initial upload status
        initial_upload_status = {
            "session_id": session_id,
            "upload_status": "received",
            "message": "File uploaded successfully. Starting processing..."
        }
        upload_status_path = session_dir / "upload_status.json"
        with open(upload_status_path, "w") as f:
            json.dump(initial_upload_status, f)

        # Create initial dataset_info.json with processing status
        initial_info = {
            "session_id": session_id,
            "processing_status": "waiting",
            "message": "Waiting for upload processing to complete..."
        }
        info_path = session_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(initial_info, f)

        # Start upload handler subprocess
        try:
            import subprocess
            handler_script = Path(__file__).parent / "upload_handler.py"

            # Create log file for subprocess output
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"upload_handler_{session_id}.log"

            with open(log_file, 'w') as logfile:
                subprocess.Popen([
                    sys.executable,
                    str(handler_script),
                    session_id,
                    str(zip_path)
                ], stdout=logfile, stderr=logfile, text=True)

            print(f"Started upload handler subprocess for session {session_id} (logs: {log_file})")
        except Exception as subprocess_error:
            print(f"Warning: Failed to start upload handler subprocess: {subprocess_error}")
            # Update status and continue anyway
            initial_upload_status["upload_status"] = "failed"
            initial_upload_status["message"] = f"Failed to start processing: {str(subprocess_error)}"
            with open(upload_status_path, "w") as f:
                json.dump(initial_upload_status, f)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded zip file {file.filename}. Processing in background.",
                "session_id": session_id,
                "file": file.filename,
                "size": len(content),
                "upload_directory": str(session_dir),
                "upload_status": "received",
                "processing_status": "waiting"
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

# Endpoint to check upload and processing status
@app.get("/upload-status/{session_id}")
async def upload_status(session_id: str):
    """Check the upload and processing status of a session."""
    session_dir = TEMP_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    # Get upload status
    upload_status_path = session_dir / "upload_status.json"
    upload_info = {
        "session_id": session_id,
        "upload_status": "unknown",
        "message": "Upload status unknown"
    }

    if upload_status_path.exists():
        try:
            with open(upload_status_path, "r") as f:
                upload_info = json.load(f)
        except Exception as e:
            upload_info = {
                "session_id": session_id,
                "upload_status": "error",
                "message": f"Failed to read upload status: {str(e)}"
            }

    # Get dataset processing status
    info_path = session_dir / "dataset_info.json"
    dataset_info = {
        "processing_status": "not_started",
        "message": "Dataset processing has not been initiated."
    }

    if info_path.exists():
        try:
            with open(info_path, "r") as f:
                dataset_info = json.load(f)
        except Exception as e:
            dataset_info = {
                "processing_status": "error",
                "error": f"Failed to read processing status: {str(e)}"
            }

    # Combine both statuses
    combined_status = {
        **upload_info,
        **dataset_info,
        "session_id": session_id
    }

    return combined_status

# Endpoint to check dataset processing status (legacy endpoint for backward compatibility)
@app.get("/dataset-status/{session_id}")
async def dataset_status(session_id: str):
    """Check the processing status of a dataset."""
    session_dir = TEMP_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    info_path = session_dir / "dataset_info.json"
    if not info_path.exists():
        return {
            "session_id": session_id,
            "processing_status": "not_started",
            "message": "Dataset processing has not been initiated."
        }

    try:
        with open(info_path, "r") as f:
            result = json.load(f)

        # Ensure processing_status is included
        if "processing_status" not in result:
            result["processing_status"] = "unknown"

        return result
    except Exception as e:
        return {
            "session_id": session_id,
            "processing_status": "error",
            "error": f"Failed to read status: {str(e)}"
        }

# Endpoint to upload processed dataset to S3 (without starting training)
@app.post("/upload-to-s3/{session_id}")
async def upload_to_s3(session_id: str):
    """Upload the processed dataset to S3 bucket using subprocess."""
    session_dir = TEMP_DIR / session_id
    
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Check if dataset is processed
    info_path = session_dir / "dataset_info.json"
    if not info_path.exists():
        raise HTTPException(status_code=400, detail="Dataset not processed yet. Please wait for processing to complete.")

    # Check if already uploaded
    s3_status_path = session_dir / "s3_upload_status.json"
    if s3_status_path.exists():
        try:
            with open(s3_status_path, 'r') as f:
                s3_status = json.load(f)
            if s3_status.get("s3_upload_status") == "completed":
                return {
                    "message": "Dataset already uploaded to cloud storage",
                    "session_id": session_id,
                    "s3_location": s3_status.get("s3_location", ""),
                    "dataset_name": s3_status.get("dataset_name", "")
                }
        except Exception:
            pass

    try:
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)

        if dataset_info.get("processing_status") != "completed":
            raise HTTPException(status_code=400, detail="Dataset processing not completed yet.")

        # Create initial S3 upload status
        initial_s3_status = {
            "session_id": session_id,
            "s3_upload_status": "starting",
            "message": "Starting S3 upload process..."
        }
        with open(s3_status_path, "w") as f:
            json.dump(initial_s3_status, f)

        # Start S3 upload subprocess
        try:
            import subprocess
            uploader_script = Path(__file__).parent / "s3_uploader_with_progress.py"

            # Create log file for subprocess output
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"s3_uploader_{session_id}.log"

            with open(log_file, 'w') as logfile:
                subprocess.Popen([
                    sys.executable,
                    str(uploader_script),
                    session_id
                ], stdout=logfile, stderr=logfile, text=True)

            print(f"Started S3 upload subprocess for session {session_id} (logs: {log_file})")

            return {
                "message": "S3 upload started in background",
                "session_id": session_id,
                "status": "uploading"
            }

        except Exception as subprocess_error:
            error_msg = f"Failed to start S3 upload subprocess: {str(subprocess_error)}"
            print(error_msg)
            
            # Update status to failed
            initial_s3_status["s3_upload_status"] = "failed"
            initial_s3_status["message"] = error_msg
            with open(s3_status_path, "w") as f:
                json.dump(initial_s3_status, f)
            
            raise HTTPException(status_code=500, detail=error_msg)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

# Endpoint to get S3 upload progress
@app.get("/upload-progress/{session_id}")
async def get_upload_progress(session_id: str):
    """Get the current S3 upload progress for a session."""
    session_dir = TEMP_DIR / session_id
    s3_status_path = session_dir / "s3_upload_status.json"
    
    if not s3_status_path.exists():
        raise HTTPException(status_code=404, detail=f"No upload progress found for session: {session_id}")
    
    try:
        with open(s3_status_path, 'r') as f:
            status_data = json.load(f)
        
        return {
            "session_id": session_id,
            "status": status_data.get("s3_upload_status", "unknown"),
            "message": status_data.get("message", ""),
            "progress": status_data.get("progress", {}),
            "s3_location": status_data.get("s3_location", ""),
            "dataset_name": status_data.get("dataset_name", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload progress: {str(e)}")

# Endpoint to check S3 upload status
@app.get("/s3-upload-status/{session_id}")
async def s3_upload_status(session_id: str):
    """Check the S3 upload status of a session."""
    session_dir = TEMP_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    s3_status_path = session_dir / "s3_upload_status.json"
    if not s3_status_path.exists():
        return {
            "session_id": session_id,
            "s3_upload_status": "not_started",
            "message": "S3 upload has not been initiated."
        }

    try:
        with open(s3_status_path, "r") as f:
            result = json.load(f)
        return result
    except Exception as e:
        return {
            "session_id": session_id,
            "s3_upload_status": "error",
            "error": f"Failed to read S3 upload status: {str(e)}"
        }

# Endpoint to manually trigger dataset processing (for recovery)
@app.post("/process-dataset/{session_id}")
async def process_dataset_manual(session_id: str):
    """Manually trigger dataset processing for a session."""
    session_dir = TEMP_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    zip_files = list(session_dir.glob("*.zip"))
    if not zip_files:
        raise HTTPException(status_code=400, detail="No zip file found for this session.")

    try:
        import subprocess
        processor_script = Path(__file__).parent / "dataset_processor.py"

        # Create log file for subprocess output
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"dataset_processor_{session_id}.log"

        with open(log_file, 'w') as logfile:
            subprocess.Popen([
                sys.executable,
                str(processor_script),
                session_id
            ], stdout=logfile, stderr=logfile, text=True)

        print(f"Manually started dataset processing subprocess for session {session_id} (logs: {log_file})")

        return {
            "message": f"Dataset processing started for session {session_id}",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

# Endpoint to get debug logs for a session
@app.get("/debug/logs/{session_id}")
async def get_debug_logs(session_id: str, lines: int = 100):
    """Get debug logs for a specific session."""
    try:
        from logger import get_session_logs
        logs = get_session_logs(session_id, lines)

        return {
            "session_id": session_id,
            "logs": logs,
            "total_lines": len(logs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

# Endpoint to get all available log files
@app.get("/debug/log-files")
async def get_log_files():
    """Get list of all available log files."""
    try:
        log_dir = Path(__file__).parent / "logs"
        if not log_dir.exists():
            return {"log_files": []}

        log_files = []
        for log_file in log_dir.glob("*.log"):
            try:
                stat = log_file.stat()
                log_files.append({
                    "filename": log_file.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "session_id": log_file.stem.split('_')[-1] if '_' in log_file.stem else None,
                    "type": log_file.stem.split('_')[0] if '_' in log_file.stem else log_file.stem
                })
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")

        # Sort by modification time (newest first)
        log_files.sort(key=lambda x: x["modified"], reverse=True)

        return {"log_files": log_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list log files: {str(e)}")

# Endpoint to clean up old log files
@app.post("/debug/cleanup")
async def cleanup_logs(days: int = 7):
    """Clean up log files older than specified days."""
    try:
        from logger import cleanup_old_logs
        cleanup_old_logs(days)
        return {"message": f"Cleaned up log files older than {days} days"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup logs: {str(e)}")

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
            'epochs': 10,
            'batch_size': 32,
            'zip_s3_path': aws_helper.s3_path,
            'session_id': session_id
        }
        output_path = f"s3://{bucket}/curate/output/"
        estimator = aws_helper.start_sagemaker_executor(
            instance_type="ml.g5.4xlarge",
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


@app.get("/download-model/{session_id}")
async def download_model(session_id: str, format: str = None):
    """Download the trained model for a completed training session."""
    try:
        available_formats = {}

        # Try to find models in S3 first (for SageMaker trained models)
        aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
        s3_client = aws_helper.s3_client
        bucket = aws_helper.bucket

        # Look for model files in the session's output directory
        prefix = f"curate/output/{session_id}/"
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if response.get('Contents'):
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.keras'):
                    available_formats['keras'] = key
                elif key.endswith('.onnx'):
                    available_formats['onnx'] = key
                elif key.endswith('.h5'):
                    available_formats['h5'] = key
                elif 'saved_model.pb' in key:
                    available_formats['saved_model'] = key

        # If not found in S3, check local temp directory (for local training)
        if not available_formats:
            session_dir = TEMP_DIR / session_id
            if session_dir.exists():
                for file_path in session_dir.glob('**/*'):
                    if file_path.is_file():
                        if file_path.name.endswith('.keras'):
                            available_formats['keras'] = str(file_path)
                        elif file_path.name.endswith('.onnx'):
                            available_formats['onnx'] = str(file_path)
                        elif file_path.name.endswith('.h5'):
                            available_formats['h5'] = str(file_path)
                        elif 'saved_model.pb' in str(file_path):
                            available_formats['saved_model'] = str(file_path)

        if not available_formats:
            raise HTTPException(status_code=404, detail="No models found for this session")

        # If no specific format requested, return info about available formats
        if format is None:
            return {
                "available_formats": list(available_formats.keys()),
                "message": "Use /download-model/{session_id}?format={keras|onnx|h5} to download specific format"
            }

        # Handle specific format request
        if format not in available_formats:
            raise HTTPException(status_code=404, detail=f"Format '{format}' not available. Available: {list(available_formats.keys())}")

        file_key = available_formats[format]

        # Handle S3 files
        if file_key.startswith('curate/output/'):
            import boto3
            s3_client = boto3.client('s3', region_name=aws_helper.s3_client.meta.region_name)
            presigned_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': file_key},
                ExpiresIn=3600  # 1 hour
            )

            return {
                "download_url": presigned_url,
                "filename": file_key.split('/')[-1],
                "format": format
            }

        # Handle local files
        else:
            return FileResponse(
                path=file_key,
                filename=file_key.split('/')[-1] if '/' in file_key else file_key.split('\\')[-1],
                media_type='application/octet-stream'
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")
