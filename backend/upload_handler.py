#!/usr/bin/env python3
"""
Upload Handler Subprocess
Handles file upload, extraction, and dataset processing initiation in a separate process.
"""

import sys
import os
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any
from logger import setup_logger

def update_upload_status(session_id: str, status: str, message: str = "", **kwargs):
    """Update the upload status file."""
    temp_dir = Path(__file__).parent / "temp_uploads"
    session_dir = temp_dir / session_id
    status_file = session_dir / "upload_status.json"

    status_data = {
        "session_id": session_id,
        "upload_status": status,
        "message": message,
        **kwargs
    }

    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)

    print(f"Upload status updated: {status} - {message}")

def handle_upload(session_id: str, zip_path: str):
    """Handle the complete upload and processing workflow."""
    logger = setup_logger("upload_handler", session_id)
    logger.info(f"Starting upload handler for session {session_id}")
    logger.debug(f"Zip path: {zip_path}")

    # Setup paths
    temp_dir = Path(__file__).parent / "temp_uploads"
    session_dir = temp_dir / session_id
    zip_file_path = Path(zip_path)

    logger.debug(f"Session directory: {session_dir}")
    logger.debug(f"Zip file exists: {zip_file_path.exists()}")

    if not zip_file_path.exists():
        error_msg = f"Zip file not found: {zip_path}"
        logger.error(error_msg)
        update_upload_status(session_id, "failed", error_msg)
        return False

    try:
        # Step 1: Update status to extracting
        logger.info("Starting zip file extraction")
        update_upload_status(session_id, "extracting", "Extracting uploaded zip file...")

        # Unzip the file
        logger.debug(f"Extracting zip file to {session_dir}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_names = zip_ref.namelist()
            logger.debug(f"Zip contains {len(zip_names)} files: {zip_names[:10]}...")
            zip_ref.extractall(session_dir)

        logger.info("Zip extraction completed")

        # Remove __MACOSX folder if present
        macosx_dir = session_dir / "__MACOSX"
        if macosx_dir.exists() and macosx_dir.is_dir():
            logger.info("Removing __MACOSX folder")
            shutil.rmtree(macosx_dir)
        else:
            logger.debug("__MACOSX folder not found")

        # Verify extraction
        extracted_files = list(session_dir.glob("*"))
        logger.debug(f"Extracted files: {[f.name for f in extracted_files]}")
        zip_files = [f for f in extracted_files if f.is_file() and f.suffix.lower() == '.zip']

        if not zip_files:
            error_msg = "No zip file found in session directory after extraction"
            logger.error(error_msg)
            update_upload_status(session_id, "failed", error_msg)
            return False

        logger.info(f"Extraction successful - {len(extracted_files)} files extracted")
        update_upload_status(session_id, "extracted", "Zip file extracted successfully", extracted_files=len(extracted_files))

        # Step 2: Start dataset processing subprocess
        logger.info("Starting dataset processing subprocess")
        update_upload_status(session_id, "processing", "Starting dataset processing...")

        try:
            import subprocess
            processor_script = Path(__file__).parent / "dataset_processor.py"

            # Create log file for subprocess output
            log_dir = Path(__file__).parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"dataset_processor_{session_id}.log"
            logger.debug(f"Dataset processor log file: {log_file}")

            with open(log_file, 'w') as logfile:
                subprocess.Popen([
                    sys.executable,
                    str(processor_script),
                    session_id
                ], stdout=logfile, stderr=logfile, text=True)

            logger.info(f"Dataset processing subprocess started successfully")
            print(f"Started dataset processing subprocess for session {session_id} (logs: {log_file})")

            update_upload_status(session_id, "processing_started", "Dataset processing started in background")
            logger.info("Upload handler completed successfully")
            return True

        except Exception as subprocess_error:
            error_msg = f"Failed to start dataset processing: {str(subprocess_error)}"
            logger.error(error_msg)
            print(f"Warning: {error_msg}")
            update_upload_status(session_id, "failed", error_msg)
            return False

    except zipfile.BadZipFile:
        error_msg = "Invalid or corrupted zip file"
        logger.error(error_msg)
        update_upload_status(session_id, "failed", error_msg)
        return False
    except Exception as e:
        error_msg = f"Upload processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Upload handling error: {e}")
        update_upload_status(session_id, "failed", error_msg)
        return False

def main():
    """Main entry point for subprocess."""
    if len(sys.argv) != 3:
        print("Usage: python upload_handler.py <session_id> <zip_path>")
        sys.exit(1)

    session_id = sys.argv[1]
    zip_path = sys.argv[2]

    success = handle_upload(session_id, zip_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
