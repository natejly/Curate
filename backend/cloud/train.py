#!/usr/bin/env python3
"""
SageMaker Training Script for Image Classification
"""

import argparse
import logging
import os
import sys

# Add directories to Python path for SageMaker environment
sys.path.insert(0, os.path.dirname(__file__))  # Add current directory (cloud)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # Add backend directory  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # Add repository root

print(f"Python path: {sys.path[:5]}")  # Debug: Show first 5 path entries
print(f"Current working directory: {os.getcwd()}")  # Debug: Show current directory

# Suppress TensorFlow verbose output
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings and info

# Suppress TensorFlow memory allocation warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# Configure TensorFlow memory growth to avoid allocation warnings
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError:
    # Memory growth must be set before GPUs have been initialized
    pass

print(f"Using GPU:{gpus}" if gpus else "No GPU found, using CPU")
# log gpu

from ImgClass.ImgClassData import ImgClassData
from ImgClass.ImgClassTrain import ImgClassTrainer
from trainio import (
    download_and_unzip, 
    print_dir_structure, 
    parse_s3_path,
    save_model,
    save_training_log,
    setup_model_directory
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Use the root logger so all messages (including from other modules) can propagate
logger = logging.getLogger()
logger.info(f"Using GPU:{gpus}" if gpus else "No GPU found, using CPU")
try:
    from advisor import TrainingAdvisor, create_advisor_summary
    AI_ADVISOR_AVAILABLE = True
except ImportError:
    AI_ADVISOR_AVAILABLE = False
    logger.warning("AI Advisor not available. Install openai package to enable: pip install openai")


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="SageMaker Image Classification Training Script")
    parser.add_argument('--zip_s3_path', type=str, 
                       default="s3://curate-sagemaker-bucket-123456789012/curate/datasets/sorted_digits_fast.zip", 
                       help="S3 path to dataset zip file")
    parser.add_argument('--session_id', type=str, default=None, help="Session ID for tracking (optional)")
    parser.add_argument('--extract_to', type=str, default="/opt/ml/input/data/train", 
                       help="Local path to extract dataset")
    parser.add_argument('--model_dir', type=str, default="/opt/ml/model", 
                       help="Directory to save trained model")
    parser.add_argument('--base_model_name', type=str, default="EfficientNetB0", 
                       help="Base model architecture to use")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--img_size', type=int, nargs=2, default=[32, 32], 
                       help="Input image size (height width)")
    parser.add_argument('--unfreeze_percent', type=float, default=0.3, 
                       help="Percentage of layers to unfreeze for fine-tuning")
    parser.add_argument('--dual_stage', action='store_true', help="Enable dual-stage training")
    parser.add_argument('--use_ai_advisor', action='store_true', 
                       help="Use AI advisor for hyperparameter optimization")
    parser.add_argument('--save_recommendations', action='store_true',
                       help="Save AI recommendations to file")
    parser.add_argument('--apply_recommendations', action='store_true',
                       help="Automatically apply AI recommendations (default: False, just display)")
    return parser.parse_args()


def setup_cloudwatch_logging(session_id):
    """Set up CloudWatch logging using boto3 directly, with region detection."""
    import boto3
    import json
    from datetime import datetime
    import os as _os
    
    log_group = "/curate/training"
    log_stream = session_id or "default-stream"
    
    try:
        region = _os.environ.get('AWS_REGION') or _os.environ.get('AWS_DEFAULT_REGION') or boto3.session.Session().region_name or 'us-east-1'
        print(f"[DEBUG] Setting up CloudWatch logging with region: {region}")
        logs_client = boto3.client('logs', region_name=region)
        
        # Create log group if it doesn't exist
        try:
            logs_client.create_log_group(logGroupName=log_group)
        except logs_client.exceptions.ResourceAlreadyExistsException:
            pass
        
        # Create log stream if it doesn't exist
        try:
            logs_client.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
        except logs_client.exceptions.ResourceAlreadyExistsException:
            pass
        
        class CloudWatchHandler(logging.Handler):
            def __init__(self, log_group, log_stream):
                super().__init__()
                self.log_group = log_group
                self.log_stream = log_stream
                self.logs_client = boto3.client('logs', region_name=region)
                self.sequence_token = None
                self.buffer = []
                self.max_buffer_size = 1  # Send logs immediately for real-time streaming
                
            def emit(self, record):
                try:
                    log_message = self.format(record)
                    timestamp = int(datetime.now().timestamp() * 1000)
                    
                    log_event = {
                        'timestamp': timestamp,
                        'message': log_message
                    }
                    
                    self.buffer.append(log_event)
                    
                    # Send immediately for real-time streaming
                    if len(self.buffer) >= self.max_buffer_size:
                        self.flush_buffer()
                        
                except Exception as e:
                    print(f"CloudWatch logging error: {e}")
                    
            def flush_buffer(self):
                if not self.buffer:
                    return
                    
                try:
                    kwargs = {
                        'logGroupName': self.log_group,
                        'logStreamName': self.log_stream,
                        'logEvents': self.buffer
                    }
                    
                    if self.sequence_token:
                        kwargs['sequenceToken'] = self.sequence_token
                    
                    response = self.logs_client.put_log_events(**kwargs)
                    self.sequence_token = response.get('nextSequenceToken')
                    self.buffer = []
                    
                except Exception as e:
                    print(f"CloudWatch buffer flush error: {e}")
                    # Reset buffer to prevent memory buildup
                    self.buffer = []
                    
            def close(self):
                self.flush_buffer()
                super().close()
        
        # Set up the custom handler
        cw_handler = CloudWatchHandler(log_group, log_stream)
        cw_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        cw_handler.setFormatter(formatter)
        
        return cw_handler

    except Exception as e:
        print(f"Failed to setup CloudWatch logging: {e}")
        return None


def handle_ai_advisor_workflow(args, data_parser, trainer, logger):
    """
    Handle the complete AI advisor workflow including recommendations and application.

    Args:
        args: Command line arguments
        data_parser: Data parser instance
        trainer: Trainer instance
        logger: Logger instance
    """
    if not args.use_ai_advisor:
        logger.info("AI Advisor not requested, using provided configuration")
        return

    if not AI_ADVISOR_AVAILABLE:
        logger.warning("AI Advisor requested but not available. Install openai package: pip install openai")
        return

    logger.info("=== AI ADVISOR: Analyzing dataset and optimizing hyperparameters ===")

    try:
        advisor = TrainingAdvisor()
        recommendations = advisor.get_hyperparameter_recommendations(data_parser, trainer)

        if not recommendations:
            logger.error("Failed to get recommendations from AI advisor")
            return

        _process_recommendations(advisor, recommendations, args, trainer, logger)

    except Exception as e:
        logger.error(f"AI Advisor error: {str(e)}")
        logger.info("Continuing with original configuration...")


def _process_recommendations(advisor, recommendations, args, trainer, logger):
    """Process and handle AI recommendations."""
    # Store original config for comparison
    original_config = advisor.get_current_config(trainer)

    # Display and log recommendations
    _display_recommendations(recommendations, logger)
    ai_log_data = advisor.format_recommendations_for_logging(recommendations, original_config)

    # Save recommendations if requested
    if args.save_recommendations:
        _save_recommendations(advisor, recommendations, args, logger)

    # Apply recommendations if requested
    changes_applied = {}
    if args.apply_recommendations:
        changes_applied = _apply_recommendations(advisor, trainer, recommendations, original_config, logger)
        ai_log_data["applied_changes"] = changes_applied
        ai_log_data["recommendation_summary"]["recommendations_applied"] = len(changes_applied)
    else:
        _log_recommendations_not_applied(logger)
        ai_log_data["applied_changes"] = {}
        ai_log_data["recommendation_summary"]["recommendations_applied"] = 0

    # Store AI recommendations in trainer for logging
    trainer.set_ai_recommendations(ai_log_data)
    logger.info("AI recommendations and reasoning stored for training log")


def _display_recommendations(recommendations, logger):
    """Display AI recommendations summary."""
    summary = create_advisor_summary(recommendations)
    logger.info(f"\n{summary}")


def _save_recommendations(advisor, recommendations, args, logger):
    """Save AI recommendations to file."""
    model_dir = setup_model_directory(args)
    rec_path = advisor.save_recommendations(
        recommendations,
        os.path.join(model_dir, 'ai_recommendations.json')
    )
    logger.info(f"AI recommendations saved to: {rec_path}")


def _apply_recommendations(advisor, trainer, recommendations, original_config, logger):
    """
    Apply AI recommendations and return applied changes.

    Returns:
        dict: Dictionary of applied changes
    """
    logger.info("Applying AI recommendations to trainer configuration...")

    # Debug logging
    _log_recommendation_structure(recommendations, logger)

    if not advisor.apply_recommendations(trainer, recommendations):
        logger.warning("Failed to apply some AI recommendations")
        return {}

    logger.info("Successfully applied AI recommendations")

    # Track and log configuration changes
    changes_applied = _track_config_changes(advisor, trainer, original_config, logger)

    # Rebuild trainer components if necessary
    _rebuild_trainer_if_needed(trainer, original_config, advisor.get_current_config(trainer), logger)

    return changes_applied


def _log_recommendation_structure(recommendations, logger):
    """Log the structure of recommendations for debugging."""
    logger.info(f"Raw recommendations keys: {list(recommendations.keys())}")

    if "hyperparameters" not in recommendations:
        return

    hyperparams = recommendations["hyperparameters"]
    logger.info(f"Hyperparameters keys: {list(hyperparams.keys())}")

    if "training_config" in hyperparams:
        training_config = hyperparams["training_config"]
        logger.info(f"Training config keys: {list(training_config.keys())}")


def _track_config_changes(advisor, trainer, original_config, logger):
    """Track and log configuration changes."""
    updated_config = advisor.get_current_config(trainer)
    changes_applied = {}

    logger.info("Configuration changes applied:")

    for key, new_value in updated_config.items():
        original_value = original_config.get(key)

        if original_value is None:
            # New configuration parameter
            logger.info(f"  {key}: {new_value} (new)")
            changes_applied[key] = {
                "original": None,
                "applied": new_value,
                "source": "ai_recommendation"
            }
        elif original_value != new_value:
            # Changed configuration parameter
            logger.info(f"  {key}: {original_value} -> {new_value}")
            changes_applied[key] = {
                "original": original_value,
                "applied": new_value,
                "source": "ai_recommendation"
            }

    return changes_applied


def _rebuild_trainer_if_needed(trainer, original_config, updated_config, logger):
    """Rebuild trainer components if image size or model changed."""
    needs_rebuild = (
        original_config.get('img_size_used') != updated_config.get('img_size_used') or
        original_config.get('base_model_name') != updated_config.get('base_model_name')
    )

    if needs_rebuild:
        logger.info("Image size or model changed, rebuilding trainer components...")
        trainer.build_datasets()
        trainer.build()


def _log_recommendations_not_applied(logger):
    """Log information when recommendations are not applied."""
    logger.info("AI recommendations generated but not applied (use --apply_recommendations to apply)")
    logger.info("You can review the recommendations above and manually adjust your configuration")


def main():
    """Main training function."""
    try:
        args = parse_args()
        
        # Set up custom CloudWatch logging
        cw_handler = setup_cloudwatch_logging(args.session_id)
        if cw_handler:
            logger.addHandler(cw_handler)
            logger.setLevel(logging.INFO)
            logger.info(f"Custom CloudWatch logging started for session {args.session_id}")
            print(f"[DEBUG] Custom CloudWatch logging is ACTIVE")
        else:
            logger.warning("CloudWatch logging setup failed, using default logging")
            logger.warning("watchtower not installed; using default SageMaker logs only")
            print(f"[DEBUG] Custom CloudWatch logging FAILED, using SageMaker default logs")
        import sys as _sys
        class _StreamToLogger:

            def __init__(self, level):

                self.level = level

            def write(self, message):

                msg = message.rstrip()

                if msg:

                    logging.getLogger().log(self.level, msg)

            def flush(self):

                pass

        _sys.stdout = _StreamToLogger(logging.INFO)

        _sys.stderr = _StreamToLogger(logging.ERROR)
        # Logging is already configured and will capture training progress
        
        logger.info("Starting SageMaker training job")
        logger.info(f"Arguments: {vars(args)}")
        
        # Download and extract dataset
        logger.info("=== DOWNLOADING DATASET ===")
        bucket, key = parse_s3_path(args.zip_s3_path)
        logger.info(f"Downloading from s3://{bucket}/{key}")
        dataset_path = download_and_unzip(bucket, key, args.extract_to)
        logger.info(f"Dataset extracted to: {dataset_path}")
        # print_dir_structure(dataset_path)
        
        # Initialize data parser
        logger.info("=== INITIALIZING DATA PARSER ===")
        data_parser = ImgClassData(dataset_path)
        logger.info(f"Found {len(data_parser.classes)} classes: {data_parser.classes}")
        
        # Initialize trainer
        logger.info("=== INITIALIZING TRAINER ===")
        trainer = ImgClassTrainer(
            dataset_path=dataset_path,
            base_model_name=args.base_model_name,
            batch_size=args.batch_size,
            initial_learning_rate=args.learning_rate,
            initial_epochs=args.epochs,
            dual_stage=args.dual_stage,
            unfreeze_percent=args.unfreeze_percent
        )
        logger.info(f"Using model: {args.base_model_name}, batch_size: {args.batch_size}, epochs: {args.epochs}")

        # AI Advisor Integration
        handle_ai_advisor_workflow(args, data_parser, trainer, logger)

        # Start training
        logger.info("=== STARTING TRAINING ===")
        logger.info(f"Training will run for {args.epochs} epochs with batch size {args.batch_size}")
        trainer.run()
        logger.info("=== TRAINING COMPLETED ===")
        trainer.training_log.show()
        
        # Setup model directory and save outputs
        logger.info("=== SAVING MODEL AND LOGS ===")
        model_dir = setup_model_directory(args)
        logger.info(f"Model directory: {model_dir}")
        
        # Save training log first (independent of model saving)
        save_training_log(trainer, model_dir, getattr(args, 'session_id', None))
        logger.info("Training log saved")
        ds_name = args.zip_s3_path.split("/")[-1].replace(".zip", "")
        # Then save the model
        save_model(trainer, model_dir, ds_name)
        logger.info("Model saved")

        # Upload models to S3 for export functionality
        logger.info("=== UPLOADING MODELS TO S3 ===")
        try:
            import subprocess
            import sys
            import os

            # Get the path to s3_uploader.py
            uploader_path = os.path.join(os.path.dirname(__file__), '..', 's3_uploader.py')

            logger.info(f"Uploading models to S3 using: {uploader_path}")
            result = subprocess.run([
                sys.executable, uploader_path, args.session_id, "--models"
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

            if result.returncode == 0:
                logger.info("Models uploaded to S3 successfully")
                logger.info(result.stdout)
            else:
                logger.warning(f"Failed to upload models to S3 (exit code: {result.returncode})")
                logger.warning(f"STDOUT: {result.stdout}")
                logger.warning(f"STDERR: {result.stderr}")
                # Don't fail the training if upload fails
        except Exception as upload_error:
            logger.warning(f"Model upload to S3 failed: {str(upload_error)}")
            # Don't fail the training if upload fails

        logger.info("=== TRAINING JOB COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"=== TRAINING FAILED ===")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    main()