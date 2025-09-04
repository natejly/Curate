#!/usr/bin/env python3
"""
Dataset Processing Subprocess
Handles dataset extraction, analysis, and metadata generation in a separate process.
"""

import sys
import os
import json
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import openai
from cloud.ImgClass.ImgClassData import ImgClassData
from cloud.aws import AWSHelper
from logger import setup_logger

def process_dataset(session_id: str):
    """Process a dataset in a separate subprocess."""
    logger = setup_logger("dataset_processor", session_id)
    logger.info(f"Starting dataset processing for session {session_id}")

    # Load environment variables
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    logger.debug(f"OpenAI API key loaded: {'Yes' if OPENAI_API_KEY else 'No'}")

    # Setup paths
    temp_dir = Path(__file__).parent / "temp_uploads"
    session_dir = temp_dir / session_id
    logger.debug(f"Session directory: {session_dir}")

    if not session_dir.exists():
        error_msg = f"Session directory {session_dir} does not exist"
        logger.error(error_msg)
        print(error_msg)
        return False

    # Find the zip file
    zip_files = list(session_dir.glob("*.zip"))
    logger.debug(f"Found {len(zip_files)} zip files in session directory")
    if not zip_files:
        error_msg = f"No zip file found in {session_dir}"
        logger.error(error_msg)
        print(error_msg)
        return False

    zip_path = zip_files[0]
    logger.info(f"Processing zip file: {zip_path}")
    logger.debug(f"Zip file size: {zip_path.stat().st_size} bytes")

    try:
        # Unzip the file (if not already done)
        logger.info("Starting zip extraction")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_names = zip_ref.namelist()
            logger.debug(f"Zip contains {len(zip_names)} files")
            zip_ref.extractall(session_dir)

        logger.info("Zip extraction completed")

        # Remove __MACOSX folder if present
        macosx_dir = session_dir / "__MACOSX"
        if macosx_dir.exists() and macosx_dir.is_dir():
            logger.info("Removing __MACOSX folder")
            import shutil
            shutil.rmtree(macosx_dir)

        # Find the first folder (assume dataset root), ignore __MACOSX
        items = [item for item in session_dir.iterdir() if item.is_dir() and item.name != "__MACOSX"]
        logger.debug(f"Found {len(items)} directories after extraction")
        if not items:
            error_msg = "No dataset folder found after unzip"
            logger.error(error_msg)
            print(error_msg)
            return False

        dataset_root = str(items[0])
        logger.info(f"Dataset root: {dataset_root}")

        # Process with ImgClassData
        logger.info("Processing dataset with ImgClassData")
        img_data = ImgClassData(dataset_root)
        logger.debug(f"Train folders: {len(img_data.train_folders)}")
        logger.debug(f"Validation folders: {len(img_data.val_folders)}")
        logger.debug(f"Test folders: {len(img_data.test_folders)}")

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

        logger.info(f"Dataset stats: {total_images} total images ({train_total} train, {val_total} val, {test_total} test)")
        logger.debug(f"Train per class: {train_per_class}")
        logger.debug(f"Validation per class: {val_per_class}")
        logger.debug(f"Test per class: {test_per_class}")
        print(f"Dataset stats: {total_images} total images ({train_total} train, {val_total} val, {test_total} test)")

        # Get LLM-inferred task
        llm_task = "NONE"
        if OPENAI_API_KEY:
            logger.info("Starting OpenAI task inference")
            try:
                file_tree = img_data.json_tree
                logger.debug(f"File tree length: {len(file_tree)} characters")
                full_prompt = f"""
                You are a data science assistant. Here is the file tree of a dataset:
                {file_tree}

                Based only on the file tree, extract the most likely ML task.
                If the task is Image Classification return "Image Classification"
                If the task is Image Segmentation return "Image Segmentation"
                If the task is Object Detection return "Object Detection"
                If the task is Text Classification return "Text Classification"
                If the task is not any of the above return "NONE"
                Only return one of these exact strings. Be strict.
                """

                openai.api_key = OPENAI_API_KEY
                logger.debug("Making OpenAI API call")
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.1
                )
                llm_task = response.choices[0].message.content.strip()
                logger.info(f"LLM inferred task: {llm_task}")
                print(f"LLM inferred task: {llm_task}")
            except Exception as e:
                error_msg = f"LLM error: {e}"
                logger.error(error_msg, exc_info=True)
                print(error_msg)
                llm_task = "ERROR"
        else:
            warning_msg = "Warning: OPENAI_API_KEY not set, skipping task inference"
            logger.warning(warning_msg)
            print(warning_msg)

        # Prepare result
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
            "task": llm_task,
            "processing_status": "completed"
        }

        logger.info("Dataset processing completed successfully")
        logger.debug(f"Classes found: {img_data.classes}")

        # Save dataset info
        info_path = session_dir / "dataset_info.json"
        logger.debug(f"Saving dataset info to: {info_path}")
        with open(info_path, "w") as f:
            json.dump(result, f, indent=2)

        # Upload dataset to S3
        try:
            logger.info("Starting S3 upload")
            aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")
            dataset_name = os.path.basename(dataset_root)
            logger.debug(f"Uploading dataset '{dataset_name}' to S3")
            print(f"Uploading dataset {dataset_name} to S3...")
            aws_helper.upload_zip(dataset_root, "curate/datasets/")
            logger.info("Dataset uploaded to S3 successfully")
            print("Dataset uploaded to S3 successfully")
        except Exception as upload_error:
            error_msg = f"Failed to upload dataset to S3: {upload_error}"
            logger.error(error_msg, exc_info=True)
            print(f"Warning: {error_msg}")
            # Update status but don't fail the entire process
            result["s3_upload_error"] = str(upload_error)
            with open(info_path, "w") as f:
                json.dump(result, f, indent=2)

        print(f"Dataset processing completed for session {session_id}")
        return True

    except Exception as e:
        error_msg = f"Error processing dataset: {e}"
        logger.error(error_msg, exc_info=True)
        print(error_msg)
        # Save error status
        error_result = {
            "session_id": session_id,
            "processing_status": "failed",
            "error": str(e)
        }
        info_path = session_dir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(error_result, f, indent=2)
        return False

def main():
    """Main entry point for subprocess."""
    if len(sys.argv) != 2:
        print("Usage: python dataset_processor.py <session_id>")
        sys.exit(1)

    session_id = sys.argv[1]
    success = process_dataset(session_id)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
