#!/usr/bin/env python3
"""
S3 Upload Subprocess
Handles dataset upload to S3 in a separate process.
"""

import sys
import os
import json
from pathlib import Path
from cloud.aws import AWSHelper
from logger import setup_logger

def update_upload_status(session_id: str, status: str, message: str = "", **kwargs):
    """Update the S3 upload status file."""
    temp_dir = Path(__file__).parent / "temp_uploads"
    session_dir = temp_dir / session_id
    status_file = session_dir / "s3_upload_status.json"

    status_data = {
        "session_id": session_id,
        "s3_upload_status": status,
        "message": message,
        **kwargs
    }

    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)

    print(f"S3 upload status updated: {status} - {message}")

def upload_dataset_to_s3(session_id: str):
    """Upload dataset to S3 in a separate subprocess."""
    logger = setup_logger("s3_uploader", session_id)
    logger.info(f"Starting dataset S3 upload for session {session_id}")

    # Setup paths
    temp_dir = Path(__file__).parent / "temp_uploads"
    session_dir = temp_dir / session_id

    if not session_dir.exists():
        error_msg = f"Session directory {session_dir} does not exist"
        logger.error(error_msg)
        update_upload_status(session_id, "failed", error_msg)
        return False

    # Check if dataset is processed
    info_path = session_dir / "dataset_info.json"
    if not info_path.exists():
        error_msg = "Dataset not processed yet"
        logger.error(error_msg)
        update_upload_status(session_id, "failed", error_msg)
        return False

    try:
        # Load dataset info
        with open(info_path, 'r') as f:
            dataset_info = json.load(f)

        if dataset_info.get("processing_status") != "completed":
            error_msg = "Dataset processing not completed yet"
            logger.error(error_msg)
            update_upload_status(session_id, "failed", error_msg)
            return False

        # Find the dataset root folder
        dataset_root = None
        logger.debug(f"Searching for dataset folder in: {session_dir}")

        for item in session_dir.iterdir():
            logger.debug(f"Found item: {item} (is_dir: {item.is_dir()}, name: {item.name})")
            if item.is_dir() and not item.name.startswith("__"):
                dataset_root = str(item)
                logger.info(f"Selected dataset root: {dataset_root}")
                break

        if not dataset_root:
            error_msg = "Dataset folder not found"
            logger.error(error_msg)
            update_upload_status(session_id, "failed", error_msg)
            return False

        # Update status to uploading
        update_upload_status(session_id, "uploading", "Uploading dataset to S3...")

        # Upload to S3
        aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
        dataset_name = Path(dataset_root).name
        s3_key = f"curate/datasets/{dataset_name}"

        logger.info(f"Starting S3 upload for dataset: {dataset_name}")
        logger.debug(f"Dataset root path: {dataset_root}")
        logger.debug(f"S3 key: {s3_key}")

        aws_helper.upload_zip(dataset_root, "curate/datasets/")

        logger.info("Dataset uploaded to S3 successfully")

        # Update status to completed
        update_upload_status(
            session_id,
            "completed",
            "Dataset uploaded to cloud storage successfully",
            s3_location=f"s3://curate-sagemaker-bucket-123456789012/{s3_key}",
            dataset_name=dataset_name
        )

        return True

    except Exception as e:
        error_msg = f"Failed to upload to S3: {str(e)}"
        logger.error(error_msg, exc_info=True)
        update_upload_status(session_id, "failed", error_msg)
        return False

def upload_models_to_s3(session_id: str):
    """Upload trained models to S3 in a separate subprocess."""
    logger = setup_logger("model_uploader", session_id)
    logger.info(f"Starting model S3 upload for session {session_id}")

    try:
        # Find the models directory
        project_root = Path(__file__).parent.parent  # Go up to Curate/
        models_dir = project_root / "curate" / "models" / session_id

        if not models_dir.exists():
            error_msg = f"Models directory {models_dir} does not exist"
            logger.error(error_msg)
            return False

        # Initialize AWS helper
        aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")

        # Upload individual model files
        uploaded_files = []

        # Upload ONNX model if it exists
        onnx_path = models_dir / "model.onnx"
        if onnx_path.exists():
            logger.info(f"Uploading ONNX model: {onnx_path}")
            s3_key = f"curate/models/{session_id}/model.onnx"
            aws_helper.upload_file(str(onnx_path), s3_key)
            uploaded_files.append(f"s3://{aws_helper.bucket}/{s3_key}")
            logger.info(f"ONNX model uploaded to: s3://{aws_helper.bucket}/{s3_key}")

        # Upload TensorFlow SavedModel directory if it exists
        tf_model_dir = models_dir / "model"
        if tf_model_dir.exists() and tf_model_dir.is_dir():
            logger.info(f"Uploading TensorFlow model directory: {tf_model_dir}")
            s3_prefix = f"curate/models/{session_id}/model"
            aws_helper.upload_directory(str(tf_model_dir), s3_prefix)
            uploaded_files.append(f"s3://{aws_helper.bucket}/{s3_prefix}/")
            logger.info(f"TensorFlow model uploaded to: s3://{aws_helper.bucket}/{s3_prefix}/")

        # Upload any other model files with dataset naming
        for file_path in models_dir.glob("*_model.*"):
            if file_path.is_file():
                filename = file_path.name
                logger.info(f"Uploading model file: {filename}")
                s3_key = f"curate/models/{session_id}/{filename}"
                aws_helper.upload_file(str(file_path), s3_key)
                uploaded_files.append(f"s3://{aws_helper.bucket}/{s3_key}")
                logger.info(f"Model file uploaded to: s3://{aws_helper.bucket}/{s3_key}")

        # Upload training log if it exists
        training_log_path = models_dir / "training_log.json"
        if training_log_path.exists():
            logger.info(f"Uploading training log: training_log.json")
            s3_key = f"curate/logs/{session_id}/training_log.json"
            aws_helper.upload_file(str(training_log_path), s3_key)
            uploaded_files.append(f"s3://{aws_helper.bucket}/{s3_key}")
            logger.info(f"Training log uploaded to: s3://{aws_helper.bucket}/{s3_key}")

        # Upload any other log files in the session directory
        for log_file in models_dir.glob("*.json"):
            if log_file.name != "training_log.json":  # Already handled above
                logger.info(f"Uploading log file: {log_file.name}")
                s3_key = f"curate/logs/{session_id}/{log_file.name}"
                aws_helper.upload_file(str(log_file), s3_key)
                uploaded_files.append(f"s3://{aws_helper.bucket}/{s3_key}")
                logger.info(f"Log file uploaded to: s3://{aws_helper.bucket}/{s3_key}")

        if uploaded_files:
            logger.info(f"Successfully uploaded {len(uploaded_files)} model files to S3")
            for file_url in uploaded_files:
                logger.info(f"  - {file_url}")
            return True
        else:
            logger.warning("No model files found to upload")
            return False

    except Exception as e:
        error_msg = f"Failed to upload models to S3: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False

# Backward compatibility
def upload_to_s3(session_id: str):
    """Backward compatibility function - uploads dataset."""
    return upload_dataset_to_s3(session_id)

def main():
    """Main entry point for subprocess."""
    if len(sys.argv) < 2:
        print("Usage: python s3_uploader.py <session_id> [--models]")
        sys.exit(1)

    session_id = sys.argv[1]

    # Check if we should upload models or datasets
    if len(sys.argv) > 2 and sys.argv[2] == "--models":
        success = upload_models_to_s3(session_id)
    else:
        success = upload_dataset_to_s3(session_id)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
