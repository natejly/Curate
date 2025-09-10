#!/usr/bin/env python3
"""
S3 Upload Subprocess with Progress Tracking
Handles dataset upload to S3 with real-time progress updates.
"""

import sys
import os
import json
import time
from pathlib import Path
from cloud.aws import AWSHelper
from logger import setup_logger
import threading

class ProgressTracker:
    """Tracks and updates upload progress."""
    
    def __init__(self, session_id: str, total_size: int = 0):
        self.session_id = session_id
        self.total_size = total_size
        self.uploaded_size = 0
        self.start_time = time.time()
        self.last_update = 0
        
    def update_progress(self, bytes_uploaded: int):
        """Update progress and write to status file."""
        self.uploaded_size += bytes_uploaded
        current_time = time.time()
        
        # Update every 0.5 seconds to avoid too many file writes
        if current_time - self.last_update >= 0.5:
            self._write_progress_status()
            self.last_update = current_time
    
    def _write_progress_status(self):
        """Write current progress to status file."""
        temp_dir = Path(__file__).parent / "temp_uploads"
        session_dir = temp_dir / self.session_id
        status_file = session_dir / "s3_upload_status.json"
        
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.uploaded_size / self.total_size * 100) if self.total_size > 0 else 0
        
        # Calculate upload speed
        speed_mbps = (self.uploaded_size / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
        
        # Estimate remaining time
        remaining_bytes = self.total_size - self.uploaded_size
        eta_seconds = (remaining_bytes / (self.uploaded_size / elapsed_time)) if self.uploaded_size > 0 and elapsed_time > 0 else 0
        
        status_data = {
            "session_id": self.session_id,
            "s3_upload_status": "uploading",
            "message": f"Uploading to cloud storage... {progress_percent:.1f}% complete",
            "progress": {
                "percent": round(progress_percent, 1),
                "uploaded_bytes": self.uploaded_size,
                "total_bytes": self.total_size,
                "speed_mbps": round(speed_mbps, 2),
                "eta_seconds": round(eta_seconds),
                "elapsed_seconds": round(elapsed_time)
            }
        }
        
        try:
            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)
            print(f"Progress: {progress_percent:.1f}% ({self.uploaded_size}/{self.total_size} bytes)")
        except Exception as e:
            print(f"Error updating progress: {e}")

def get_directory_size(directory_path: str) -> int:
    """Calculate total size of directory for progress tracking."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(file_path)
            except (OSError, IOError):
                continue
    return total_size

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

class ProgressCallback:
    """Callback class for boto3 upload progress."""
    
    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker
        
    def __call__(self, bytes_amount):
        self.tracker.update_progress(bytes_amount)

def upload_dataset_to_s3_with_progress(session_id: str):
    """Upload dataset to S3 with progress tracking."""
    logger = setup_logger("s3_uploader", session_id)
    logger.info(f"Starting dataset S3 upload with progress for session {session_id}")

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

        # Calculate total size for progress tracking
        logger.info("Calculating dataset size for progress tracking...")
        total_size = get_directory_size(dataset_root)
        logger.info(f"Total dataset size: {total_size / (1024*1024):.2f} MB")

        # Initialize progress tracker
        progress_tracker = ProgressTracker(session_id, total_size)
        
        # Update status to uploading with initial progress
        update_upload_status(
            session_id, 
            "uploading", 
            "Preparing cloud upload...",
            progress={
                "percent": 0,
                "uploaded_bytes": 0,
                "total_bytes": total_size,
                "speed_mbps": 0,
                "eta_seconds": 0,
                "elapsed_seconds": 0
            }
        )

        # Upload to S3 with progress tracking
        aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
        dataset_name = Path(dataset_root).name
        s3_key = f"curate/datasets/{dataset_name}"

        logger.info(f"Starting S3 upload for dataset: {dataset_name}")
        logger.debug(f"Dataset root path: {dataset_root}")
        logger.debug(f"S3 key: {s3_key}")

        # Use the enhanced upload method with progress callback
        progress_callback = ProgressCallback(progress_tracker)
        aws_helper.upload_zip_with_progress(dataset_root, "curate/datasets/", progress_callback)

        logger.info("Dataset uploaded to S3 successfully")

        # Update status to completed
        update_upload_status(
            session_id,
            "completed",
            "Dataset uploaded to cloud storage successfully",
            s3_location=f"s3://curate-sagemaker-bucket-123456789012/{s3_key}",
            dataset_name=dataset_name,
            progress={
                "percent": 100,
                "uploaded_bytes": total_size,
                "total_bytes": total_size,
                "speed_mbps": progress_tracker.uploaded_size / (1024 * 1024) / (time.time() - progress_tracker.start_time),
                "eta_seconds": 0,
                "elapsed_seconds": round(time.time() - progress_tracker.start_time)
            }
        )

        return True

    except Exception as e:
        error_msg = f"Failed to upload to S3: {str(e)}"
        logger.error(error_msg, exc_info=True)
        update_upload_status(session_id, "failed", error_msg)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python s3_uploader_with_progress.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    print(f"Starting S3 upload with progress for session: {session_id}")
    
    success = upload_dataset_to_s3_with_progress(session_id)
    if success:
        print("S3 upload completed successfully")
        sys.exit(0)
    else:
        print("S3 upload failed")
        sys.exit(1)

