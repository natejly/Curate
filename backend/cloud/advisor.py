"""
AI Advisor for hyperparameter optimization using OpenAI API
Analyzes dataset characteristics and generates optimized training configurations
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import openai
except ImportError:
    openai = None

from prompts import (
    HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT,
    get_hyperparameter_prompt,
    get_dataset_analysis_prompt,
    get_error_analysis_prompt,
    OPTIMIZATION_SYSTEM_PROMPT,
    get_optimization_prompt
)

logger = logging.getLogger(__name__)


class TrainingAdvisor:
    """AI-powered training advisor for hyperparameter optimization."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize the training advisor.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use for analysis
        """
        if openai is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
            
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _extract_value(self, param_dict: Dict[str, Any]):
        """
        Extract value from AI recommendation parameter dictionary.
        Expects "value" key as specified in the prompt.
        """
        if not isinstance(param_dict, dict):
            return param_dict  # If it's already a value, return it

        if "value" in param_dict:
            value = param_dict["value"]

            # Type conversion for common parameter types
            if isinstance(value, str):
                # Try to convert string numbers to appropriate numeric types
                if value.isdigit():
                    return int(value)
                try:
                    return float(value)
                except ValueError:
                    pass

            return value

        # Log available keys for debugging if "value" is missing
        logger.warning(f"Expected 'value' key not found in parameter dict. Available keys: {list(param_dict.keys())}")
        return None
        
    def extract_dataset_info(self, data_parser, trainer) -> Dict[str, Any]:
        """
        Extract relevant information from ImgClassData and ImgClassTrainer.
        
        Args:
            data_parser: ImgClassData instance
            trainer: ImgClassTrainer instance
            
        Returns:
            Dictionary containing dataset characteristics
        """
        try:
            # Extract basic dataset info
            dataset_info = {
                "dataset_path": data_parser.filepath,
                "total_images": data_parser.total_images if hasattr(data_parser, 'total_images') else "unknown",
                "num_classes": len(data_parser.classes),
                "class_names": data_parser.classes,
                "image_dimensions": {
                    "original": data_parser.IMSIZE,
                    "processed": trainer.IMG_SIZE
                },
                "file_tree_structure": data_parser.json_tree,
                "directory_structure": {
                    "train_dir": data_parser.train_dir,
                    "val_dir": data_parser.val_dir,
                    "test_dir": data_parser.test_dir
                }
            }
            
            # Calculate class distribution if possible
            try:
                class_distribution = {}
                if hasattr(data_parser, 'json_tree') and data_parser.json_tree:
                    for class_name in data_parser.classes:
                        class_path = os.path.join(data_parser.train_dir, class_name)
                        if os.path.exists(class_path):
                            class_count = len([f for f in os.listdir(class_path) 
                                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                            class_distribution[class_name] = class_count
                
                dataset_info["class_distribution"] = class_distribution
                dataset_info["total_training_images"] = sum(class_distribution.values()) if class_distribution else "unknown"
                
                # Calculate class balance metrics
                if class_distribution:
                    counts = list(class_distribution.values())
                    dataset_info["class_balance"] = {
                        "min_class_size": min(counts),
                        "max_class_size": max(counts),
                        "mean_class_size": sum(counts) / len(counts),
                        "imbalance_ratio": max(counts) / min(counts) if min(counts) > 0 else "infinite"
                    }
            except Exception as e:
                logger.warning(f"Could not calculate class distribution: {str(e)}")
                dataset_info["class_distribution"] = "calculation_failed"
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Failed to extract dataset info: {str(e)}")
            return {"error": str(e)}
    
    def get_current_config(self, trainer) -> Dict[str, Any]:
        """
        Extract current training configuration from trainer.
        
        Args:
            trainer: ImgClassTrainer instance
            
        Returns:
            Dictionary containing current configuration
        """
        try:
            return {
                "base_model_name": trainer.base_model_name,
                "batch_size": trainer.batch_size,
                "initial_learning_rate": trainer.initial_learning_rate,
                "fine_tune_learning_rate": trainer.fine_tune_learning_rate,
                "initial_epochs": trainer.initial_epochs,
                "fine_tune_epochs": trainer.fine_tune_epochs,
                "dual_stage": trainer.dual_stage,
                "custom_img_size": trainer.custom_img_size,
                "img_size_used": trainer.IMG_SIZE,
                "unfreeze_percent": trainer.unfreeze_percent,
                "num_classes": trainer.NUM_CLASSES
            }
        except Exception as e:
            logger.error(f"Failed to extract current config: {str(e)}")
            return {"error": str(e)}
    
    def call_openai_api(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Make API call to OpenAI and parse JSON response.
        
        Args:
            system_prompt: System prompt for the AI
            user_prompt: User prompt with specific request
            
        Returns:
            Parsed JSON response or None if failed
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=4000,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Raw response: {content}")
            return None
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            return None
    
    def get_hyperparameter_recommendations(self, data_parser, trainer) -> Optional[Dict[str, Any]]:
        """
        Get AI-powered hyperparameter recommendations.
        
        Args:
            data_parser: ImgClassData instance
            trainer: ImgClassTrainer instance
            
        Returns:
            Dictionary containing recommendations or None if failed
        """
        logger.info("Extracting dataset information for AI analysis...")
        dataset_info = self.extract_dataset_info(data_parser, trainer)
        current_config = self.get_current_config(trainer)
        
        logger.info("Calling OpenAI API for hyperparameter recommendations...")
        user_prompt = get_hyperparameter_prompt(
            json.dumps(dataset_info, indent=2),
            json.dumps(current_config, indent=2)
        )
        
        recommendations = self.call_openai_api(
            HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT,
            user_prompt
        )
        
        if recommendations:
            logger.info("Successfully received AI recommendations")
            return recommendations
        else:
            logger.error("Failed to get recommendations from AI advisor")
            return None
    
    def save_recommendations(self, recommendations: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Save recommendations to JSON file.
        
        Args:
            recommendations: Recommendations dictionary
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"ai_recommendations_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(recommendations, f, indent=2, default=str)
            
            logger.info(f"AI recommendations saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save recommendations: {str(e)}")
            raise
    
    def apply_recommendations(self, trainer, recommendations: Dict[str, Any]) -> bool:
        """
        Apply AI recommendations to trainer configuration.
        
        Args:
            trainer: ImgClassTrainer instance
            recommendations: AI recommendations dictionary
            
        Returns:
            True if successfully applied, False otherwise
        """
        try:
            # Debug: Log the structure of recommendations
            logger.info(f"Recommendations structure: {list(recommendations.keys())}")
            
            # Check if recommendations have the expected structure
            if "hyperparameters" in recommendations:
                params = recommendations["hyperparameters"]
                logger.info("Using 'hyperparameters' structure")
            elif "ai_recommendations" in recommendations:
                # Handle the structure from format_recommendations_for_logging
                params = recommendations["ai_recommendations"]
                logger.info("Using 'ai_recommendations' structure")
            else:
                logger.error("No hyperparameters or ai_recommendations found in recommendations")
                logger.error(f"Available keys: {list(recommendations.keys())}")
                return False
            
            logger.info(f"Params structure: {list(params.keys()) if isinstance(params, dict) else 'Not a dict'}")
            
            # Apply training config
            if "training_config" in params:
                config = params["training_config"]
                
                if "batch_size" in config:
                    value = self._extract_value(config["batch_size"])
                    logger.info(f"Raw batch_size from AI: {config['batch_size']}")
                    logger.info(f"Extracted batch_size value: {value} (type: {type(value)})")

                    # Ensure batch_size is a valid integer
                    if value:
                        try:
                            trainer.batch_size = int(value)
                            logger.info(f"Updated batch_size to: {trainer.batch_size} (type: {type(trainer.batch_size)})")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid batch_size value: {value}, keeping original value")
                            logger.error(f"Error: {e}")
                    else:
                        logger.warning("No valid batch_size value extracted from AI recommendations")
                
                if "initial_learning_rate" in config:
                    value = self._extract_value(config["initial_learning_rate"])
                    if value:
                        trainer.initial_learning_rate = value
                        logger.info(f"Updated initial_learning_rate to: {trainer.initial_learning_rate}")
                
                if "initial_epochs" in config:
                    value = self._extract_value(config["initial_epochs"])
                    if value:
                        trainer.initial_epochs = value
                        logger.info(f"Updated initial_epochs to: {trainer.initial_epochs}")
                
                if "image_size" in config:
                    value = self._extract_value(config["image_size"])
                    logger.info(f"Raw image_size from AI: {config['image_size']}")
                    logger.info(f"Extracted image_size value: {value} (type: {type(value)})")

                    # Ensure image_size is a valid tuple/list of two integers
                    if value:
                        try:
                            if isinstance(value, (list, tuple)) and len(value) == 2:
                                trainer.custom_img_size = tuple(int(x) for x in value)
                                trainer.IMG_SIZE = trainer.custom_img_size
                                logger.info(f"Updated image_size to: {trainer.IMG_SIZE}")
                            else:
                                logger.warning(f"Invalid image_size format: {value}, expected [width, height]")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid image_size value: {value}, keeping original value")
                            logger.error(f"Error: {e}")
                    else:
                        logger.warning("No valid image_size value extracted from AI recommendations")
            
            # Apply fine-tuning config
            if "fine_tuning_config" in params:
                config = params["fine_tuning_config"]
                
                if "fine_tune_learning_rate" in config:
                    value = self._extract_value(config["fine_tune_learning_rate"])
                    if value:
                        trainer.fine_tune_learning_rate = value
                        logger.info(f"Updated fine_tune_learning_rate to: {trainer.fine_tune_learning_rate}")
                
                if "fine_tune_epochs" in config:
                    value = self._extract_value(config["fine_tune_epochs"])
                    if value:
                        trainer.fine_tune_epochs = value
                        logger.info(f"Updated fine_tune_epochs to: {trainer.fine_tune_epochs}")
                
                if "unfreeze_percent" in config:
                    value = self._extract_value(config["unfreeze_percent"])
                    if value:
                        trainer.unfreeze_percent = value
                        logger.info(f"Updated unfreeze_percent to: {trainer.unfreeze_percent}")
            
            # Apply model architecture
            if "model_architecture" in params:
                model_config = params["model_architecture"]
                if "base_model" in model_config:
                    value = self._extract_value(model_config["base_model"])
                    if value:
                        trainer.base_model_name = value
                        logger.info(f"Updated base_model_name to: {trainer.base_model_name}")
                elif "recommended_model" in model_config:
                    trainer.base_model_name = model_config["recommended_model"]
                    logger.info(f"Updated base_model_name to: {trainer.base_model_name}")
            
            # Apply dual-stage recommendation
            if "analysis" in recommendations and "recommended_approach" in recommendations["analysis"]:
                approach = recommendations["analysis"]["recommended_approach"]
                trainer.dual_stage = (approach == "dual_stage")
                logger.info(f"Updated dual_stage to: {trainer.dual_stage}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply recommendations: {str(e)}")
            return False
    
    def optimize(self, trainer) -> Optional[Dict[str, Any]]:
        """
        Analyze training logs and suggest optimized hyperparameters for better performance.
        
        Args:
            trainer: ImgClassTrainer instance with training history
            
        Returns:
            Dictionary containing optimization recommendations or None if failed
        """
        try:
            # Extract training log data
            if not hasattr(trainer, 'training_log') or trainer.training_log is None:
                logger.error("No training log found in trainer")
                return None
            
            # Get the training history and current configuration
            training_log = trainer.training_log.get_log_data()
            current_config = self.get_current_config(trainer)
            
            logger.info("Analyzing training performance for optimization...")
            
            # Create optimization prompt
            optimization_prompt = get_optimization_prompt(
                json.dumps(training_log, indent=2),
                json.dumps(current_config, indent=2)
            )
            
            # Get optimization recommendations from AI
            recommendations = self.call_openai_api(OPTIMIZATION_SYSTEM_PROMPT, optimization_prompt)
            
            if recommendations:
                logger.info("Successfully received optimization recommendations")
                
                # Apply recommendations using trainer.edit_config to maintain training log updates
                if "optimization_recommendations" in recommendations:
                    self._apply_optimization_recommendations(trainer, recommendations["optimization_recommendations"])
                
                return recommendations
            else:
                logger.error("Failed to get optimization recommendations from AI")
                return None
                
        except Exception as e:
            logger.error(f"Failed to optimize training: {str(e)}")
            return None
    
    def _apply_optimization_recommendations(self, trainer, recommendations: Dict[str, Any]) -> bool:
        """
        Apply optimization recommendations using trainer.edit_config to maintain log updates.
        
        Args:
            trainer: ImgClassTrainer instance
            recommendations: Optimization recommendations dictionary
            
        Returns:
            True if successfully applied, False otherwise
        """
        try:
            changes_applied = {}
            
            # Collect all parameters for edit_config
            config_params = {}
            
            # Get current values as defaults
            config_params['base_model_name'] = trainer.base_model_name
            config_params['batch_size'] = trainer.batch_size
            config_params['initial_learning_rate'] = trainer.initial_learning_rate
            config_params['fine_tune_learning_rate'] = trainer.fine_tune_learning_rate
            config_params['initial_epochs'] = trainer.initial_epochs
            config_params['fine_tune_epochs'] = trainer.fine_tune_epochs
            config_params['dual_stage'] = trainer.dual_stage
            config_params['custom_img_size'] = trainer.custom_img_size
            config_params['unfreeze_percent'] = trainer.unfreeze_percent
            
            # Update with recommendations from training_config
            if "training_config" in recommendations:
                for param, details in recommendations["training_config"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        
                        # Special handling for image_size -> custom_img_size
                        if param == "image_size":
                            config_params['custom_img_size'] = tuple(new_value) if isinstance(new_value, list) else new_value
                            changes_applied['custom_img_size'] = {
                                "old_value": old_value,
                                "new_value": new_value,
                                "reasoning": details.get("reasoning", "N/A")
                            }
                        else:
                            config_params[param] = new_value
                            changes_applied[param] = {
                                "old_value": old_value,
                                "new_value": new_value,
                                "reasoning": details.get("reasoning", "N/A")
                            }
            
            # Update with recommendations from fine_tuning_config
            if "fine_tuning_config" in recommendations:
                for param, details in recommendations["fine_tuning_config"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        config_params[param] = new_value
                        changes_applied[param] = {
                            "old_value": old_value,
                            "new_value": new_value,
                            "reasoning": details.get("reasoning", "N/A")
                        }
            
            # Update with recommendations from model_architecture
            if "model_architecture" in recommendations:
                for param, details in recommendations["model_architecture"].items():
                    if isinstance(details, dict) and "recommended_value" in details:
                        old_value = config_params.get(param)
                        new_value = details["recommended_value"]
                        # Map base_model_name parameter
                        if param == "base_model_name":
                            config_params['base_model_name'] = new_value
                            changes_applied[param] = {
                                "old_value": old_value,
                                "new_value": new_value,
                                "reasoning": details.get("reasoning", "N/A")
                            }
            
            # Apply all changes at once using edit_config
            if changes_applied and hasattr(trainer, 'edit_config'):
                try:
                    trainer.edit_config(
                        base_model_name=config_params['base_model_name'],
                        batch_size=config_params['batch_size'],
                        initial_learning_rate=config_params['initial_learning_rate'],
                        fine_tune_learning_rate=config_params['fine_tune_learning_rate'],
                        initial_epochs=config_params['initial_epochs'],
                        fine_tune_epochs=config_params['fine_tune_epochs'],
                        dual_stage=config_params['dual_stage'],
                        custom_img_size=config_params['custom_img_size'],
                        unfreeze_percent=config_params['unfreeze_percent']
                    )
                    
                    # Log all applied changes
                    for param, change_info in changes_applied.items():
                        logger.info(f"Applied optimization: {param} = {change_info['new_value']} (was {change_info['old_value']})")
                    
                    logger.info(f"Successfully applied {len(changes_applied)} optimization changes using edit_config")
                    return True
                    
                except Exception as edit_error:
                    logger.error(f"Failed to apply optimizations using edit_config: {str(edit_error)}")
                    # Fallback to direct assignment
                    for param, change_info in changes_applied.items():
                        try:
                            setattr(trainer, param, change_info['new_value'])
                            logger.info(f"Applied optimization (fallback): {param} = {change_info['new_value']}")
                        except Exception as fallback_error:
                            logger.warning(f"Failed to apply {param}: {str(fallback_error)}")
                    return True
            else:
                logger.info("No optimization changes to apply")
                return False
            
        except Exception as e:
            logger.error(f"Failed to apply optimization recommendations: {str(e)}")
            return False

    def format_recommendations_for_logging(self, recommendations: Dict[str, Any], original_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format AI recommendations for inclusion in training logs.
        
        Args:
            recommendations: Full AI recommendations
            original_config: Original trainer configuration
            
        Returns:
            Formatted recommendations for logging
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model,
                "dataset_analysis": recommendations.get("analysis", {}),
                "original_configuration": original_config,
                "ai_recommendations": {},
                "applied_changes": {},
                "recommendation_summary": {}
            }
            
            # Extract hyperparameter recommendations with reasoning
            if "hyperparameters" in recommendations:
                params = recommendations["hyperparameters"]
                
                # Training config recommendations
                if "training_config" in params:
                    log_entry["ai_recommendations"]["training_config"] = {}
                    for param, details in params["training_config"].items():
                        if isinstance(details, dict) and "value" in details:
                            log_entry["ai_recommendations"]["training_config"][param] = {
                                "recommended_value": details["value"],
                                "confidence": details.get("confidence", "N/A"),
                                "reasoning": details.get("reasoning", "No reasoning provided")
                            }
                
                # Fine-tuning config recommendations
                if "fine_tuning_config" in params:
                    log_entry["ai_recommendations"]["fine_tuning_config"] = {}
                    for param, details in params["fine_tuning_config"].items():
                        if isinstance(details, dict) and "value" in details:
                            log_entry["ai_recommendations"]["fine_tuning_config"][param] = {
                                "recommended_value": details["value"],
                                "confidence": details.get("confidence", "N/A"),
                                "reasoning": details.get("reasoning", "No reasoning provided")
                            }
                
                # Model architecture recommendations
                if "model_architecture" in params:
                    arch = params["model_architecture"]
                    if isinstance(arch, dict):
                        log_entry["ai_recommendations"]["model_architecture"] = {
                            "recommended_model": arch.get("base_model", "N/A"),
                            "confidence": arch.get("confidence", "N/A"),
                            "reasoning": arch.get("reasoning", "No reasoning provided")
                        }
                
                # Optimization recommendations
                if "optimization" in params:
                    opt = params["optimization"]
                    log_entry["ai_recommendations"]["optimization"] = {}
                    for opt_type, details in opt.items():
                        if isinstance(details, dict):
                            log_entry["ai_recommendations"]["optimization"][opt_type] = {
                                "recommended_settings": details,
                                "confidence": details.get("confidence", "N/A"),
                                "reasoning": details.get("reasoning", "No reasoning provided")
                            }
            
            # SageMaker recommendations
            if "sagemaker_recommendations" in recommendations:
                sm_rec = recommendations["sagemaker_recommendations"]
                log_entry["ai_recommendations"]["sagemaker"] = {}
                for rec_type, details in sm_rec.items():
                    if isinstance(details, dict) and "value" in details:
                        log_entry["ai_recommendations"]["sagemaker"][rec_type] = {
                            "recommended_value": details["value"],
                            "confidence": details.get("confidence", "N/A"),
                            "reasoning": details.get("reasoning", "No reasoning provided")
                        }
                    else:
                        log_entry["ai_recommendations"]["sagemaker"][rec_type] = details
            
            # Data augmentation recommendations
            if "data_augmentation" in recommendations:
                aug_rec = recommendations["data_augmentation"]
                log_entry["ai_recommendations"]["data_augmentation"] = aug_rec
            
            # Applied changes will be added by the caller
            log_entry["applied_changes"] = {}
            
            # Create summary statistics
            total_recommendations = 0
            
            for category in log_entry["ai_recommendations"].values():
                if isinstance(category, dict):
                    total_recommendations += len(category)
            
            log_entry["recommendation_summary"] = {
                "total_parameters_analyzed": len(original_config),
                "total_recommendations_made": total_recommendations,
                "ai_model_used": self.model,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Failed to format recommendations for logging: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "raw_recommendations": recommendations
            }


def create_advisor_summary(recommendations: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of AI recommendations.
    
    Args:
        recommendations: AI recommendations dictionary
        
    Returns:
        Formatted summary string
    """
    try:
        summary = "=== AI ADVISOR RECOMMENDATIONS ===\n\n"
        
        if "analysis" in recommendations:
            analysis = recommendations["analysis"]
            summary += f"Dataset Complexity: {analysis.get('dataset_complexity', 'Unknown')}\n"
            summary += f"Recommended Approach: {analysis.get('recommended_approach', 'Unknown')}\n\n"
            
            if "key_insights" in analysis:
                summary += "Key Insights:\n"
                for insight in analysis["key_insights"]:
                    summary += f"  - {insight}\n"
                summary += "\n"
        
        if "hyperparameters" in recommendations:
            params = recommendations["hyperparameters"]
            summary += "RECOMMENDED HYPERPARAMETERS:\n\n"
            
            if "training_config" in params:
                config = params["training_config"]
                summary += "Training Configuration:\n"
                for key, value in config.items():
                    if isinstance(value, dict) and "value" in value:
                        conf = value.get("confidence", "N/A")
                        summary += f"  {key}: {value['value']} (confidence: {conf}%)\n"
                summary += "\n"
            
            if "fine_tuning_config" in params:
                config = params["fine_tuning_config"]
                summary += "Fine-tuning Configuration:\n"
                for key, value in config.items():
                    if isinstance(value, dict) and "value" in value:
                        conf = value.get("confidence", "N/A")
                        summary += f"  {key}: {value['value']} (confidence: {conf}%)\n"
                summary += "\n"
        
        if "sagemaker_recommendations" in recommendations:
            sm_rec = recommendations["sagemaker_recommendations"]
            summary += "SageMaker Recommendations:\n"
            if "instance_type" in sm_rec:
                summary += f"  Instance Type: {sm_rec['instance_type']['value']}\n"
            if "estimated_training_time" in sm_rec:
                summary += f"  Estimated Training Time: {sm_rec['estimated_training_time'].get('dual_stage', 'N/A')}\n"
            if "cost_estimate" in sm_rec:
                summary += f"  Estimated Cost: {sm_rec['cost_estimate'].get('approximate_cost', 'N/A')}\n"
        
        return summary
        
    except Exception as e:
        return f"Error creating summary: {str(e)}"
    

