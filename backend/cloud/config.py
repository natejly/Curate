"""
Configuration management for training pipeline.
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    
    # Dataset configuration
    zip_s3_path: str = "s3://curate-sagemaker-bucket-123456789012/curate/datasets/sorted_digits_fast.zip"
    extract_to: str = "/opt/ml/input/data/train"
    session_id: Optional[str] = None
    
    # Model configuration  
    base_model_name: str = "EfficientNetB0"
    img_size: Tuple[int, int] = (32, 32)
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 5
    dual_stage: bool = False
    unfreeze_percent: float = 0.3
    
    # Output configuration
    model_dir: str = "/opt/ml/model"
    
    # Optimization configuration
    early_stop_threshold: float = 0.99
    max_iterations: int = 5
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not (0.0 <= self.unfreeze_percent <= 1.0):
            raise ValueError("unfreeze_percent must be between 0.0 and 1.0")
        if not (0.0 <= self.early_stop_threshold <= 1.0):
            raise ValueError("early_stop_threshold must be between 0.0 and 1.0")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if len(self.img_size) != 2 or any(s <= 0 for s in self.img_size):
            raise ValueError("img_size must be a tuple of two positive integers")


class ConfigManager:
    """Manages training configuration from command line arguments."""
    
    @staticmethod
    def parse_args() -> TrainingConfig:
        """Parse command line arguments and return TrainingConfig."""
        parser = argparse.ArgumentParser(
            description="SageMaker Image Classification Training Script"
        )
        
        # Dataset arguments
        parser.add_argument('--zip_s3_path', type=str, 
                           default=TrainingConfig.zip_s3_path,
                           help="S3 path to dataset zip file")
        parser.add_argument('--session_id', type=str, default=None, 
                           help="Session ID for tracking (optional)")
        parser.add_argument('--extract_to', type=str, 
                           default=TrainingConfig.extract_to,
                           help="Local path to extract dataset")
        
        # Model arguments
        parser.add_argument('--model_dir', type=str, 
                           default=TrainingConfig.model_dir,
                           help="Directory to save trained model")
        parser.add_argument('--base_model_name', type=str, 
                           default=TrainingConfig.base_model_name,
                           help="Base model architecture to use")
        parser.add_argument('--img_size', type=int, nargs=2, 
                           default=list(TrainingConfig.img_size),
                           help="Input image size (height width)")
        
        # Training arguments
        parser.add_argument('--batch_size', type=int, 
                           default=TrainingConfig.batch_size,
                           help="Training batch size")
        parser.add_argument('--learning_rate', type=float, 
                           default=TrainingConfig.learning_rate,
                           help="Initial learning rate")
        parser.add_argument('--epochs', type=int, 
                           default=TrainingConfig.epochs,
                           help="Number of training epochs")
        parser.add_argument('--unfreeze_percent', type=float, 
                           default=TrainingConfig.unfreeze_percent,
                           help="Percentage of layers to unfreeze for fine-tuning")
        parser.add_argument('--dual_stage', action='store_true', 
                           help="Enable dual-stage training")
        
        # Optimization arguments
        parser.add_argument('--early_stop_threshold', type=float, 
                           default=TrainingConfig.early_stop_threshold,
                           help="Test accuracy threshold for early stopping")
        parser.add_argument('--max_iterations', type=int, 
                           default=TrainingConfig.max_iterations,
                           help="Maximum number of optimization iterations")
        
        args = parser.parse_args()
        
        # Create config from parsed arguments
        config = TrainingConfig(
            zip_s3_path=args.zip_s3_path,
            extract_to=args.extract_to,
            session_id=args.session_id,
            base_model_name=args.base_model_name,
            img_size=tuple(args.img_size),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            dual_stage=args.dual_stage,
            unfreeze_percent=args.unfreeze_percent,
            model_dir=args.model_dir,
            early_stop_threshold=args.early_stop_threshold,
            max_iterations=args.max_iterations
        )
        
        # Validate configuration
        config.validate()
        
        return config
