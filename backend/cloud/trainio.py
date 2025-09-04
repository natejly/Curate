"""
Training I/O utilities for SageMaker Image Classification
Handles data loading, model saving, logging, and S3 operations
"""

import boto3
import zipfile
import os
import logging
import json
import numpy as np
import tensorflow as tf
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TensorFlowJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for TensorFlow and NumPy objects."""
    def default(self, obj):
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        return super().default(obj)


def convert_to_serializable(obj):
    """Convert TensorFlow/NumPy objects to JSON-serializable format."""
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist() if obj.shape != () else obj.numpy().item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist() if obj.shape != () else obj.item()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    else:
        return obj


def safe_json_dump(data, filepath):
    """Safely dump data to JSON file with TensorFlow/NumPy support."""
    try:
        safe_data = convert_to_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(safe_data, f, cls=TensorFlowJSONEncoder, indent=2)
        logger.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save JSON to {filepath}: {str(e)}")


def download_and_unzip(bucket, key, extract_to="/opt/ml/input/data/train"):
    """Download and extract dataset from S3."""
    try:
        s3 = boto3.client("s3")
        os.makedirs(extract_to, exist_ok=True)
        local_zip = "/tmp/dataset.zip"
        logger.info(f"Downloading s3://{bucket}/{key}")
        s3.download_file(bucket, key, local_zip)
        logger.info(f"Extracting {local_zip} to {extract_to}")
        with zipfile.ZipFile(local_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(local_zip)
        logger.info("Extraction complete")
        return extract_to
    except Exception as e:
        logger.error(f"Failed to download/extract dataset: {str(e)}")
        raise


def print_dir_structure(path, max_depth=3):
    """Print directory structure for debugging."""
    logger.info(f"Directory structure for {path}:")
    try:
        for root, dirs, files in os.walk(path):
            level = root.replace(path, "").count(os.sep)
            if level >= max_depth:
                dirs[:] = []
                continue
            indent = "  " * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            subindent = "  " * (level + 1)
            for i, f in enumerate(files[:5]):
                logger.info(f"{subindent}{f}")
            if len(files) > 5:
                logger.info(f"{subindent}... and {len(files) - 5} more files")
    except Exception as e:
        logger.warning(f"Could not print directory structure: {str(e)}")


def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key components."""
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")
    path = s3_path[5:]
    if "/" not in path:
        raise ValueError("S3 path must contain bucket and key")
    bucket, key = path.split("/", 1)
    return bucket, key


def save_training_log(trainer, model_dir):
    """Save training log to model directory with error handling."""
    training_log_path = os.path.join(model_dir, 'training_log.json')
    logger.info(f"Saving training log to: {training_log_path}")
    
    try:
        trainer.training_log.save(training_log_path)
        
        # Verify training log was saved
        if os.path.exists(training_log_path):
            log_size = os.path.getsize(training_log_path)
            logger.info(f"Training log saved successfully: {training_log_path} ({log_size} bytes)")
            
            # Show a preview of the log content
            with open(training_log_path, 'r') as f:
                content_preview = f.read()[:200]
                logger.info(f"Training log preview: {content_preview}...")
        else:
            logger.warning(f"Training log was not saved to: {training_log_path}")
            
    except Exception as log_error:
        logger.error(f"Failed to save training log: {str(log_error)}")
        # Try to save with a basic approach as fallback
        try:
            log_data = trainer.training_log.getLog()
            with open(training_log_path, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            logger.info(f"Training log saved with fallback method: {training_log_path}")
        except Exception as fallback_error:
            logger.error(f"Fallback training log save also failed: {str(fallback_error)}")


def save_model_with_tensor_fix(trainer, model_dir):
    """Save model with TensorFlow compatibility fixes for SageMaker."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        was_eager = tf.executing_eagerly()
        if was_eager:
            logger.info("Temporarily disabling eager execution for model saving")
            
        # Use SageMaker-compatible numeric directory name for serving
        save_attempts = [('00000001', 'tf'), ('model.h5', 'h5'), ('model.keras', None)]
        
        for filename, save_format in save_attempts:
            model_path = os.path.join(model_dir, filename)
            try:
                logger.info(f"Attempting to save model to {model_path} (format: {save_format or 'keras'})")
                tf.keras.backend.clear_session()
                
                if save_format == 'tf':
                    tf.saved_model.save(trainer.model, model_path)
                elif save_format:
                    trainer.model.save(model_path, save_format=save_format)
                else:
                    trainer.model.save(model_path)
                    
                logger.info("Model saved successfully")
                
                # Log file/directory size
                if os.path.isfile(model_path):
                    file_size = os.path.getsize(model_path) / (1024 * 1024)
                    logger.info(f"Model file size: {file_size:.2f} MB")
                elif os.path.isdir(model_path):
                    total_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                                   for dirpath, dirnames, filenames in os.walk(model_path) 
                                   for filename in filenames)
                    dir_size = total_size / (1024 * 1024)
                    logger.info(f"Model directory size: {dir_size:.2f} MB")
                    
                return
                
            except Exception as format_error:
                logger.warning(f"Failed to save in {save_format or 'keras'} format: {str(format_error)}")
                if os.path.exists(model_path):
                    if os.path.isfile(model_path):
                        os.remove(model_path)
                    elif os.path.isdir(model_path):
                        shutil.rmtree(model_path)
                continue
                
        # If all standard formats failed, try weights only
        logger.info("All standard formats failed, attempting to save weights only")
        weights_path = os.path.join(model_dir, 'model_weights.h5')
        trainer.model.save_weights(weights_path)
        
        # Save architecture separately
        architecture_path = os.path.join(model_dir, 'model_architecture.json')
        try:
            model_json = trainer.model.to_json()
            with open(architecture_path, 'w') as f:
                f.write(model_json)
        except Exception as json_error:
            logger.warning(f"Could not save model architecture as JSON: {str(json_error)}")
            
        logger.info("Model weights and architecture saved successfully")
        weights_size = os.path.getsize(weights_path) / (1024 * 1024)
        logger.info(f"Model weights size: {weights_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save model in any format: {str(e)}")
        raise


def save_model(trainer, model_dir):
    """Main model saving function."""
    save_model_with_tensor_fix(trainer, model_dir)


def setup_model_directory(args):
    """Setup and verify model directory from environment or args."""
    model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"SM_MODEL_DIR env var: {os.environ.get('SM_MODEL_DIR', 'NOT SET')}")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Model directory created/verified: {model_dir}")
    
    return model_dir