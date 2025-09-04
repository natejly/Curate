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
    """Save model in both TensorFlow (.keras) and ONNX formats."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        was_eager = tf.executing_eagerly()
        if was_eager:
            logger.info("Temporarily disabling eager execution for model saving")

        # Save in TensorFlow format first
        keras_path = os.path.join(model_dir, 'model.keras')
        logger.info(f"Saving TensorFlow model to {keras_path}")
        tf.keras.backend.clear_session()
        trainer.model.save(keras_path)

        # Log TensorFlow model size
        keras_size = os.path.getsize(keras_path) / (1024 * 1024)
        logger.info(f"TensorFlow model saved: {keras_size:.2f} MB")

        # Attempt ONNX export
        try:
            export_model_to_onnx(trainer.model, model_dir)
            onnx_path = os.path.join(model_dir, 'model.onnx')
            if os.path.exists(onnx_path):
                onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
                logger.info(f"ONNX model exported: {onnx_size:.2f} MB")
        except Exception as onnx_error:
            logger.warning(f"ONNX export failed (TensorFlow model still available): {str(onnx_error)}")

        logger.info("Model saving completed - both formats attempted")
        return

    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}")
        raise e


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


def export_model_to_onnx(model, model_dir):
    """Export TensorFlow/Keras model to ONNX format."""
    try:
        # Try to import tf2onnx
        import tf2onnx

        onnx_path = os.path.join(model_dir, 'model.onnx')
        logger.info("Attempting ONNX export...")

        # Convert model to ONNX
        # Note: This requires input signature specification
        # For simplicity, we'll use a basic conversion approach

        # Create a representative input shape based on model input
        input_signature = None
        if hasattr(model, 'input_shape') and model.input_shape:
            # Remove batch dimension for ONNX
            input_shape = [dim if dim is not None else 1 for dim in model.input_shape[1:]]
            input_signature = [tf.TensorSpec([None] + input_shape, tf.float32, name='input')]

        if input_signature:
            # Convert to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=13  # ONNX opset version
            )

            # Save ONNX model
            with open(onnx_path, 'wb') as f:
                f.write(model_proto.SerializeToString())

            # Log file size
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"ONNX model exported successfully: {file_size:.2f} MB")
        else:
            logger.warning("Could not determine input signature for ONNX conversion")
            raise ValueError("Unable to create input signature for ONNX export")

    except ImportError:
        logger.info("tf2onnx not installed - skipping ONNX export. Install with: pip install tf2onnx")
        raise ImportError("tf2onnx not available")
    except Exception as e:
        logger.warning(f"ONNX export failed: {str(e)}")
        raise e
