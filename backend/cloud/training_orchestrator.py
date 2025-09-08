"""
Training orchestrator for managing the complete training pipeline.
"""

import logging
import subprocess
import sys
import os
from typing import Optional

from config import TrainingConfig
from logging_setup import LoggingManager
from ai_workflow import AIWorkflowManager
from ImgClass.ImgClassData import ImgClassData
from ImgClass.ImgClassTrain import ImgClassTrainer
from trainio import (
    download_and_unzip, 
    parse_s3_path,
    save_model,
    save_training_log,
    setup_model_directory
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates the complete training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = None
        self.ai_workflow = AIWorkflowManager(config)
        self.data_parser = None
        self.trainer = None
    
    def setup_logging(self) -> None:
        """Setup complete logging configuration."""
        self.logger = LoggingManager.setup_complete_logging(self.config.session_id)
        logger.info("Logging setup completed")
    
    def setup_dataset(self) -> str:
        """Download and setup dataset."""
        logger.info("=== DOWNLOADING DATASET ===")
        bucket, key = parse_s3_path(self.config.zip_s3_path)
        logger.info(f"Downloading from s3://{bucket}/{key}")
        
        dataset_path = download_and_unzip(bucket, key, self.config.extract_to)
        logger.info(f"Dataset extracted to: {dataset_path}")
        
        return dataset_path
    
    def initialize_data_parser(self, dataset_path: str) -> None:
        """Initialize data parser."""
        logger.info("=== INITIALIZING DATA PARSER ===")
        self.data_parser = ImgClassData(dataset_path)
        logger.info(f"Found {len(self.data_parser.classes)} classes: {self.data_parser.classes}")
    
    def initialize_trainer(self, dataset_path: str) -> None:
        """Initialize trainer with configuration."""
        logger.info("=== INITIALIZING TRAINER ===")
        
        self.trainer = ImgClassTrainer(
            dataset_path=dataset_path,
            base_model_name=self.config.base_model_name,
            batch_size=self.config.batch_size,
            initial_learning_rate=self.config.learning_rate,
            initial_epochs=self.config.epochs,
            dual_stage=self.config.dual_stage,
            unfreeze_percent=self.config.unfreeze_percent,
            custom_img_size=self.config.img_size,
            early_stop_threshold=self.config.early_stop_threshold,
            max_iterations=self.config.max_iterations
        )
        
        logger.info(f"Using model: {self.config.base_model_name}, "
                   f"batch_size: {self.config.batch_size}, "
                   f"epochs: {self.config.epochs}, "
                   f"img_size: {self.config.img_size}")
        logger.info(f"Early stopping threshold: {self.config.early_stop_threshold * 100:.1f}%")
        logger.info(f"Max optimization iterations: {self.config.max_iterations}")
    
    def run_initial_ai_recommendations(self) -> None:
        """Run initial AI recommendations workflow."""
        if not self.ai_workflow.is_available():
            logger.info("AI Advisor not available, skipping initial recommendations")
            return
        
        self.ai_workflow.get_initial_recommendations(self.data_parser, self.trainer)
    
    def run_training(self) -> None:
        """Run the main training process."""
        logger.info("=== STARTING TRAINING ===")
        logger.info(f"Training will run for {self.config.epochs} epochs with batch size {self.config.batch_size}")
        
        self.trainer.run()
        
        logger.info("=== INITIAL TRAINING COMPLETED ===")
    
    def run_optimization_iterations(self) -> None:
        """Run AI optimization iterations."""
        success = self.ai_workflow.run_optimization_iterations(self.trainer)
        
        if success:
            logger.info("=== OPTIMIZATION COMPLETED ===")
        else:
            logger.info("=== OPTIMIZATION ENDED EARLY OR FAILED ===")
        
        # Show final training log
        self.trainer.training_log.show()
    
    def save_outputs(self) -> None:
        """Save model and training logs."""
        logger.info("=== SAVING MODEL AND LOGS ===")
        
        model_dir = setup_model_directory(self.config)
        logger.info(f"Model directory: {model_dir}")
        
        # Save training log first (independent of model saving)
        save_training_log(self.trainer, model_dir, self.config.session_id)
        logger.info("Training log saved")
        
        # Extract dataset name from S3 path
        ds_name = self.config.zip_s3_path.split("/")[-1].replace(".zip", "")
        
        # Save the model
        save_model(self.trainer, model_dir, ds_name)
        logger.info("Model saved")
    
    def upload_models_to_s3(self) -> None:
        """Upload models to S3 for export functionality."""
        if not self.config.session_id:
            logger.info("No session_id provided, skipping S3 upload")
            return
        
        logger.info("=== UPLOADING MODELS TO S3 ===")
        
        try:
            # Get the path to s3_uploader.py
            uploader_path = os.path.join(os.path.dirname(__file__), '..', 's3_uploader.py')
            
            logger.info(f"Uploading models to S3 using: {uploader_path}")
            result = subprocess.run([
                sys.executable, uploader_path, self.config.session_id, "--models"
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
    
    def run_complete_pipeline(self) -> None:
        """Run the complete training pipeline."""
        try:
            logger.info("Starting SageMaker training job")
            logger.info(f"Configuration: {vars(self.config)}")
            
            # Setup dataset
            dataset_path = self.setup_dataset()
            
            # Initialize components
            self.initialize_data_parser(dataset_path)
            self.initialize_trainer(dataset_path)
            
            # Run AI advisor initial recommendations
            self.run_initial_ai_recommendations()
            
            # Run main training
            self.run_training()
            
            # Run optimization iterations
            self.run_optimization_iterations()
            
            # Save outputs
            self.save_outputs()
            
            # Upload to S3
            self.upload_models_to_s3()
            
            logger.info("=== TRAINING JOB COMPLETED SUCCESSFULLY ===")
            
        except Exception as e:
            logger.error("=== TRAINING FAILED ===")
            logger.error(f"Error: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def run_training_pipeline(config: Optional[TrainingConfig] = None) -> None:
    """
    Run the complete training pipeline.
    
    Args:
        config: Training configuration. If None, will be parsed from command line.
    """
    if config is None:
        from config import ConfigManager
        config = ConfigManager.parse_args()
    
    # Create orchestrator and setup logging
    orchestrator = TrainingOrchestrator(config)
    orchestrator.setup_logging()
    
    # Run the complete pipeline
    orchestrator.run_complete_pipeline()
