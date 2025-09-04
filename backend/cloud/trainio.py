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
        # Handle TensorFlow tensors (including EagerTensor)
        if isinstance(obj, tf.Tensor):
            tensor_data = obj.numpy()
            if tensor_data.shape == ():  # scalar tensor
                return tensor_data.item()
            else:  # multi-dimensional tensor
                return tensor_data.tolist()
        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            if obj.shape == ():  # scalar array
                return obj.item()
            else:  # multi-dimensional array
                return obj.tolist()
        # Handle NumPy scalar types
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()
        # Handle objects with __dict__ attribute (fallback)
        elif hasattr(obj, '__dict__'):
            return {k: self.default(v) for k, v in obj.__dict__.items()}
        return super().default(obj)


def convert_to_serializable(obj):
    """Convert TensorFlow/NumPy objects to JSON-serializable format."""
    if isinstance(obj, tf.Tensor):
        tensor_data = obj.numpy()
        if tensor_data.shape == ():  # scalar tensor
            return tensor_data.item()
        else:  # multi-dimensional tensor
            return tensor_data.tolist()
    elif isinstance(obj, np.ndarray):
        if obj.shape == ():  # scalar array
            return obj.item()
        else:  # multi-dimensional array
            return obj.tolist()
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


def save_model(trainer, model_dir):
    """Save model in TensorFlow format and upload to S3."""
    try:
        os.makedirs(model_dir, exist_ok=True)

        # Save in TensorFlow format
        keras_path = os.path.join(model_dir, 'model.keras')
        logger.info(f"Saving TensorFlow model to {keras_path}")

        # Clear session to avoid memory issues
        tf.keras.backend.clear_session()

        # Save the model
        trainer.model.save(keras_path)

        # Log model size
        keras_size = os.path.getsize(keras_path) / (1024 * 1024)
        logger.info(f"TensorFlow model saved: {keras_size:.2f} MB")

        return keras_path

    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}")
        raise e


def upload_model_to_s3(model_path, bucket_name="curate-sagemaker-bucket-123456789012", s3_prefix="curate/models/"):
    """Upload saved model to S3 bucket."""
    try:
        import boto3
        from datetime import datetime

        # Initialize S3 client
        s3_client = boto3.client('s3')

        # Generate S3 key with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.basename(model_path)
        s3_key = f"{s3_prefix}{timestamp}_{model_filename}"

        # Upload the model file
        logger.info(f"Uploading model to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(model_path, bucket_name, s3_key)

        # Verify upload
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model uploaded successfully: {model_size:.2f} MB")

        s3_location = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"Model available at: {s3_location}")

        return s3_location

    except Exception as e:
        logger.error(f"Failed to upload model to S3: {str(e)}")
        raise e


def save_and_upload_model(trainer, model_dir, bucket_name="curate-sagemaker-bucket-123456789012"):
    """Save model and upload to S3 in one function."""
    # Save the model
    model_path = save_model(trainer, model_dir)

    # Upload to S3
    s3_location = upload_model_to_s3(model_path, bucket_name)

    return model_path, s3_location


def export_model_to_onnx(model, model_dir, input_signature=None):
    """Export TensorFlow/Keras model to ONNX format with better error handling."""
    onnx_path = os.path.join(model_dir, 'model.onnx')

    try:
        # Try to import tf2onnx
        import tf2onnx
        logger.info("Attempting ONNX export...")

        # If no input signature provided, try to infer from model
        if input_signature is None:
            if hasattr(model, 'input_shape') and model.input_shape:
                # Remove batch dimension for ONNX and handle None dimensions
                input_shape = []
                for dim in model.input_shape[1:]:  # Skip batch dimension
                    if dim is None:
                        input_shape.append(1)  # Default to 1 for dynamic dimensions
                    else:
                        input_shape.append(dim)

                # Use common image input shape if it looks like an image model
                if len(input_shape) == 3 and input_shape[-1] in [1, 3]:  # Grayscale or RGB
                    input_signature = [tf.TensorSpec([None] + input_shape, tf.float32, name='input')]
                else:
                    # Generic input signature
                    input_signature = [tf.TensorSpec([None] + input_shape, tf.float32, name='input')]
            else:
                logger.warning("Could not determine input signature for ONNX conversion - using generic signature")
                # Fallback to a generic signature that might work
                input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input')]

        if input_signature:
            # Convert to ONNX with better error handling
            logger.info(f"Using input signature: {input_signature[0].shape}")

            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=input_signature,
                opset=13  # ONNX opset version - widely supported
            )

            # Save ONNX model
            with open(onnx_path, 'wb') as f:
                f.write(model_proto.SerializeToString())

            # Log file size
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"ONNX model exported successfully: {file_size:.2f} MB")
            return onnx_path
        else:
            logger.warning("Could not create input signature for ONNX export")
            return None

    except ImportError:
        logger.warning("tf2onnx not installed - skipping ONNX export. Install with: pip install tf2onnx")
        return None
    except Exception as e:
        logger.warning(f"ONNX export failed: {str(e)}")
        logger.info("Continuing with TensorFlow format only")
        return None


def save_model_formats(trainer, model_dir):
    """Save model in both TensorFlow (.keras) and ONNX formats."""
    try:
        os.makedirs(model_dir, exist_ok=True)

        # Clear session to avoid memory issues
        tf.keras.backend.clear_session()

        saved_files = {}

        # Save in TensorFlow format
        keras_path = os.path.join(model_dir, 'model.keras')
        logger.info(f"Saving TensorFlow model to {keras_path}")

        trainer.model.save(keras_path)

        # Log TensorFlow model size
        keras_size = os.path.getsize(keras_path) / (1024 * 1024)
        logger.info(f"TensorFlow model saved: {keras_size:.2f} MB")
        saved_files['keras'] = keras_path

        # Attempt ONNX export
        onnx_path = export_model_to_onnx(trainer.model, model_dir)
        if onnx_path:
            saved_files['onnx'] = onnx_path

        return saved_files

    except Exception as e:
        logger.error(f"Model saving failed: {str(e)}")
        raise e


def upload_models_to_s3(model_files, bucket_name="curate-sagemaker-bucket-123456789012", s3_prefix="curate/models/", session_id=None):
    """Upload multiple model files to S3 bucket."""
    try:
        import boto3
        from datetime import datetime

        # Initialize S3 client
        s3_client = boto3.client('s3')

        # Generate timestamp for this upload batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include session ID in the path if provided
        if session_id:
            session_prefix = f"{s3_prefix.rstrip('/')}/sessions/{session_id}/"
        else:
            session_prefix = s3_prefix

        uploaded_locations = {}

        for format_name, model_path in model_files.items():
            # Generate S3 key with timestamp and format
            model_filename = os.path.basename(model_path)
            if session_id:
                s3_key = f"{session_prefix}{timestamp}_{format_name}_{model_filename}"
            else:
                s3_key = f"{session_prefix}{timestamp}_{format_name}_{model_filename}"

            # Upload the model file
            logger.info(f"Uploading {format_name} model to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(model_path, bucket_name, s3_key)

            # Verify upload
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"{format_name.upper()} model uploaded successfully: {model_size:.2f} MB")

            s3_location = f"s3://{bucket_name}/{s3_key}"
            uploaded_locations[format_name] = s3_location
            logger.info(f"{format_name.upper()} model available at: {s3_location}")

        return uploaded_locations

    except Exception as e:
        logger.error(f"Failed to upload models to S3: {str(e)}")
        raise e


def fetch_model_from_s3(s3_path, download_dir="./downloads"):
    """Fetch a model file from S3 and download to local directory."""
    try:
        import boto3

        # Parse S3 path
        if not s3_path.startswith("s3://"):
            raise ValueError("S3 path must start with s3://")

        path_parts = s3_path[5:].split("/", 1)
        if len(path_parts) != 2:
            raise ValueError("Invalid S3 path format")

        bucket_name, s3_key = path_parts

        # Initialize S3 client
        s3_client = boto3.client('s3')

        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Generate local file path
        local_filename = os.path.basename(s3_key)
        local_path = os.path.join(download_dir, local_filename)

        # Download the file
        logger.info(f"Downloading {s3_path} to {local_path}")
        s3_client.download_file(bucket_name, s3_key, local_path)

        # Verify download
        file_size = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"Model downloaded successfully: {file_size:.2f} MB")

        return local_path

    except Exception as e:
        logger.error(f"Failed to fetch model from S3: {str(e)}")
        raise e


def list_session_models(bucket_name="curate-sagemaker-bucket-123456789012", session_id=None, s3_prefix="curate/models/"):
    """List all models for a specific session or all sessions."""
    try:
        import boto3

        # Initialize S3 client
        s3_client = boto3.client('s3')

        if session_id:
            # List models for specific session
            prefix = f"{s3_prefix.rstrip('/')}/sessions/{session_id}/"
        else:
            # List all sessions and their models
            prefix = f"{s3_prefix.rstrip('/')}/sessions/"

        # List objects with the session prefix
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' not in response:
            logger.info(f"No models found for session: {session_id}")
            return {}

        models_by_session = {}

        for obj in response['Contents']:
            key = obj['Key']
            size_mb = obj['Size'] / (1024 * 1024)

            # Parse session ID from path
            path_parts = key.split('/')
            if len(path_parts) >= 4 and path_parts[-4] == 'sessions':
                current_session_id = path_parts[-3]
            else:
                current_session_id = 'unknown'

            if current_session_id not in models_by_session:
                models_by_session[current_session_id] = []

            # Determine model format from filename
            filename = os.path.basename(key)
            if '_keras_' in filename:
                format_type = 'keras'
            elif '_onnx_' in filename:
                format_type = 'onnx'
            else:
                format_type = 'unknown'

            model_info = {
                's3_key': key,
                's3_path': f"s3://{bucket_name}/{key}",
                'filename': filename,
                'format': format_type,
                'size_mb': round(size_mb, 2),
                'last_modified': obj['LastModified'].isoformat()
            }

            models_by_session[current_session_id].append(model_info)

        return models_by_session

    except Exception as e:
        logger.error(f"Failed to list session models: {str(e)}")
        raise e


def fetch_session_model(session_id, format_type='keras', bucket_name="curate-sagemaker-bucket-123456789012",
                       s3_prefix="curate/models/", download_dir="./downloads"):
    """Fetch the latest model of specified format for a session."""
    try:
        # List all models for the session
        session_models = list_session_models(bucket_name, session_id, s3_prefix)

        if session_id not in session_models:
            raise ValueError(f"No models found for session: {session_id}")

        # Find the most recent model of the specified format
        models = session_models[session_id]
        format_models = [m for m in models if m['format'] == format_type]

        if not format_models:
            available_formats = set(m['format'] for m in models)
            raise ValueError(f"No {format_type} models found for session {session_id}. Available formats: {available_formats}")

        # Sort by last modified (most recent first) and get the first one
        latest_model = max(format_models, key=lambda m: m['last_modified'])

        # Fetch the model
        s3_path = latest_model['s3_path']
        logger.info(f"Fetching latest {format_type} model for session {session_id}")
        local_path = fetch_model_from_s3(s3_path, download_dir)

        return local_path, latest_model

    except Exception as e:
        logger.error(f"Failed to fetch session model: {str(e)}")
        raise e


def save_and_upload_models(trainer, model_dir, bucket_name="curate-sagemaker-bucket-123456789012", session_id=None):
    """Save model in multiple formats and upload to S3."""
    # Save models in both formats
    saved_files = save_model_formats(trainer, model_dir)

    # Upload all formats to S3
    uploaded_locations = upload_models_to_s3(saved_files, bucket_name, session_id=session_id)

    return saved_files, uploaded_locations


def setup_model_directory(args):
    """Setup and verify model directory from environment or args."""
    model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"SM_MODEL_DIR env var: {os.environ.get('SM_MODEL_DIR', 'NOT SET')}")

    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Model directory created/verified: {model_dir}")

    return model_dir


