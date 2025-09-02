#!/usr/bin/env python3
"""
SageMaker Training Script for Image Classification
"""

import argparse
import logging
import os

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

# Import AI advisor (optional)
try:
    from advisor import TrainingAdvisor, create_advisor_summary
    AI_ADVISOR_AVAILABLE = True
except ImportError:
    AI_ADVISOR_AVAILABLE = False
    logger.warning("AI Advisor not available. Install openai package to enable: pip install openai")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def main():
    """Main training function."""
    try:
        args = parse_args()
        logger.info("Starting SageMaker training job")
        logger.info(f"Arguments: {vars(args)}")
        
        # Download and extract dataset
        bucket, key = parse_s3_path(args.zip_s3_path)
        dataset_path = download_and_unzip(bucket, key, args.extract_to)
        # print_dir_structure(dataset_path)
        
        # Initialize data parser
        logger.info("Initializing data parser")
        data_parser = ImgClassData(dataset_path)
        
        # Initialize trainer
        logger.info("Initializing trainer")
        trainer = ImgClassTrainer(
            dataset_path=dataset_path,
            base_model_name=args.base_model_name,
            batch_size=args.batch_size,
            initial_learning_rate=args.learning_rate,
            initial_epochs=args.epochs,
            dual_stage=args.dual_stage,
            unfreeze_percent=args.unfreeze_percent
        )

        
        # AI Advisor Integration
        if args.use_ai_advisor and AI_ADVISOR_AVAILABLE:
            logger.info("=== AI ADVISOR: Analyzing dataset and optimizing hyperparameters ===")
            try:
                # Initialize AI advisor (will read from OPENAI_API_KEY environment variable)
                advisor = TrainingAdvisor()
                
                # Get AI recommendations
                recommendations = advisor.get_hyperparameter_recommendations(data_parser, trainer)
                
                if recommendations:
                    # Store original config for comparison
                    original_config = advisor.get_current_config(trainer)
                    
                    # Display summary
                    summary = create_advisor_summary(recommendations)
                    logger.info(f"\n{summary}")
                    
                    # Format recommendations for training log
                    ai_log_data = advisor.format_recommendations_for_logging(recommendations, original_config)
                    
                    # Save recommendations if requested
                    if args.save_recommendations:
                        model_dir = setup_model_directory(args)
                        rec_path = advisor.save_recommendations(
                            recommendations, 
                            os.path.join(model_dir, 'ai_recommendations.json')
                        )
                        logger.info(f"AI recommendations saved to: {rec_path}")
                    
                    # Apply recommendations if requested
                    if args.apply_recommendations:
                        logger.info("Applying AI recommendations to trainer configuration...")
                        
                        # Debug: Log the actual structure of recommendations
                        logger.info(f"Raw recommendations keys: {list(recommendations.keys())}")
                        if "hyperparameters" in recommendations:
                            logger.info(f"Hyperparameters keys: {list(recommendations['hyperparameters'].keys())}")
                            if "training_config" in recommendations["hyperparameters"]:
                                logger.info(f"Training config keys: {list(recommendations['hyperparameters']['training_config'].keys())}")
                        
                        if advisor.apply_recommendations(trainer, recommendations):
                            logger.info("Successfully applied AI recommendations")
                            
                            # Log the configuration changes
                            updated_config = advisor.get_current_config(trainer)
                            logger.info("Configuration changes applied:")
                            changes_applied = {}
                            for key, value in updated_config.items():
                                if key in original_config and original_config[key] != value:
                                    logger.info(f"  {key}: {original_config[key]} -> {value}")
                                    changes_applied[key] = {
                                        "original": original_config[key],
                                        "applied": value,
                                        "source": "ai_recommendation"
                                    }
                                elif key not in original_config:
                                    logger.info(f"  {key}: {value} (new)")
                                    changes_applied[key] = {
                                        "original": None,
                                        "applied": value,
                                        "source": "ai_recommendation"
                                    }
                            
                            # Update the AI log data with applied changes
                            ai_log_data["applied_changes"] = changes_applied
                            ai_log_data["recommendation_summary"]["recommendations_applied"] = len(changes_applied)
                            total_recs = ai_log_data["recommendation_summary"]["total_recommendations_made"]
                            if total_recs > 0:
                                ai_log_data["recommendation_summary"]["application_rate"] = f"{(len(changes_applied)/total_recs)*100:.1f}%"
                            
                            # Rebuild trainer components with new config if image size changed
                            if (original_config.get('img_size_used') != updated_config.get('img_size_used') or
                                original_config.get('base_model_name') != updated_config.get('base_model_name')):
                                logger.info("Image size or model changed, rebuilding trainer components...")
                                trainer.build_datasets()
                                trainer.build()
                        else:
                            logger.warning("Failed to apply some AI recommendations")
                            ai_log_data["applied_changes"] = {}
                            ai_log_data["recommendation_summary"]["recommendations_applied"] = 0
                            ai_log_data["recommendation_summary"]["application_rate"] = "0%"
                    else:
                        logger.info("AI recommendations generated but not applied (use --apply_recommendations to apply)")
                        logger.info("You can review the recommendations above and manually adjust your configuration")
                        ai_log_data["applied_changes"] = {}
                        ai_log_data["recommendation_summary"]["recommendations_applied"] = 0
                        ai_log_data["recommendation_summary"]["application_rate"] = "0%"
                    
                    # Store AI recommendations in trainer for logging
                    trainer.set_ai_recommendations(ai_log_data)
                    logger.info("AI recommendations and reasoning stored for training log")
                else:
                    logger.error("Failed to get recommendations from AI advisor")
                    
            except Exception as e:
                logger.error(f"AI Advisor error: {str(e)}")
                logger.info("Continuing with original configuration...")
                
        elif args.use_ai_advisor and not AI_ADVISOR_AVAILABLE:
            logger.warning("AI Advisor requested but not available. Install openai package: pip install openai")
        else:
            logger.info("AI Advisor not requested, using provided configuration")
        
        # Start training
        logger.info("Starting training")
        trainer.run()
        trainer.training_log.show()
        
        # Setup model directory and save outputs
        model_dir = setup_model_directory(args)
        
        # Save training log first (independent of model saving)
        save_training_log(trainer, model_dir)
        
        # Then save the model
        save_model(trainer, model_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()