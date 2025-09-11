"""
AI Advisor workflow management for training pipeline.
"""

import logging
import os
from typing import Dict, Any, Optional

from config import TrainingConfig

logger = logging.getLogger(__name__)

# Check AI Advisor availability
try:
    from advisor import TrainingAdvisor, create_advisor_summary
    AI_ADVISOR_AVAILABLE = True
except ImportError:
    AI_ADVISOR_AVAILABLE = False
    logger.warning("AI Advisor not available. Install openai package to enable: pip install openai")


class AIWorkflowManager:
    """Manages AI advisor workflow for training optimization."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.advisor = None
        
        if AI_ADVISOR_AVAILABLE:
            self.advisor = TrainingAdvisor(session_id=config.session_id)
    
    def is_available(self) -> bool:
        """Check if AI advisor is available."""
        return AI_ADVISOR_AVAILABLE and self.advisor is not None
    
    def get_initial_recommendations(self, data_parser, trainer) -> Optional[Dict[str, Any]]:
        """
        Get initial hyperparameter recommendations before training.
        
        Args:
            data_parser: ImgClassData instance
            trainer: ImgClassTrainer instance
            
        Returns:
            Recommendations dictionary or None if failed
        """
        if not self.is_available():
            logger.warning("AI Advisor not available for initial recommendations")
            return None
        
        logger.info("=== AI ADVISOR: Analyzing dataset and optimizing hyperparameters ===")
        
        try:
            recommendations = self.advisor.get_hyperparameter_recommendations(data_parser, trainer)
            
            if not recommendations:
                logger.error("Failed to get recommendations from AI advisor")
                return None
            
            self._process_recommendations(recommendations, trainer)
            return recommendations
            
        except Exception as e:
            logger.error(f"AI Advisor error: {str(e)}")
            logger.info("Continuing with original configuration...")
            return None
    
    def run_optimization_iterations(self, trainer) -> bool:
        """
        Run AI optimization iterations after initial training.
        
        Args:
            trainer: ImgClassTrainer instance
            
        Returns:
            True if optimization completed successfully, False otherwise
        """
        if not self.is_available():
            logger.info("AI Advisor not available, skipping optimization iterations")
            return False
        
        # Check if early stopping threshold already reached
        if self._should_skip_optimization(trainer):
            return True
        
        logger.info("=== STARTING AI OPTIMIZATION ITERATIONS ===")
        logger.info(f"Running up to {self.config.max_iterations} optimization iterations")
        
        try:
            return self._run_optimization_loop(trainer)
        except Exception as e:
            logger.error(f"AI optimization failed: {str(e)}")
            logger.info("Continuing with training results obtained so far...")
            return False
    
    def _should_skip_optimization(self, trainer) -> bool:
        """Check if optimization should be skipped due to early stopping."""
        if trainer.metrics and trainer.metrics.get('accuracy', 0) > self.config.early_stop_threshold:
            logger.info(f"EARLY STOPPING: Test accuracy {trainer.metrics['accuracy']:.4f} exceeds {self.config.early_stop_threshold*100:.1f}% threshold")
            logger.info("Skipping optimization iterations due to already high test accuracy")
            return True
        return False
    
    def _run_optimization_loop(self, trainer) -> bool:
        """Run the optimization iteration loop."""
        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"=== OPTIMIZATION ITERATION {iteration}/{self.config.max_iterations} ===")
            
            # Set the optimization iteration number in the trainer
            trainer.set_optimization_iteration(iteration)
            
            optimization_results = self.advisor.optimize(trainer)
            if not optimization_results:
                logger.warning(f"Optimization iteration {iteration} failed, stopping optimization process")
                return False
            
            logger.info(f"Optimization {iteration} recommendations applied. Running training with fresh model...")
            
            # Run training with optimized parameters on fresh model
            trainer.run()
            logger.info(f"=== OPTIMIZATION {iteration} TRAINING COMPLETED ===")
            
            # Log the specific test accuracy
            if trainer.metrics and 'accuracy' in trainer.metrics:
                logger.info(f"Test accuracy for iteration {iteration}: {trainer.metrics['accuracy']:.4f}")
            
            trainer.training_log.show()
            
            # Check for early stopping after each optimization iteration
            if self._check_early_stopping(trainer, iteration):
                return True
        
        return True
    
    def _check_early_stopping(self, trainer, iteration: int) -> bool:
        """Check if early stopping criteria is met."""
        if trainer.metrics and trainer.metrics.get('accuracy', 0) > self.config.early_stop_threshold:
            logger.info(f"🎯 EARLY STOPPING: Test accuracy {trainer.metrics['accuracy']:.4f} exceeds {self.config.early_stop_threshold*100:.1f}% threshold")
            logger.info(f"Stopping optimization iterations after iteration {iteration}/{self.config.max_iterations}")
            return True
        return False
    
    def _process_recommendations(self, recommendations: Dict[str, Any], trainer) -> None:
        """Process and apply AI recommendations."""
        # Store recommendations for reasoning extraction
        self._current_recommendations = recommendations
        
        # Store original config for comparison
        original_config = self.advisor.get_current_config(trainer)
        
        # Display and log recommendations
        self._display_recommendations(recommendations)
        ai_log_data = self.advisor.format_recommendations_for_logging(recommendations, original_config)
        
        # Save recommendations
        self._save_recommendations(recommendations)
        
        # Apply recommendations
        changes_applied = self._apply_recommendations(trainer, recommendations, original_config)
        ai_log_data["applied_changes"] = changes_applied
        ai_log_data["recommendation_summary"]["recommendations_applied"] = len(changes_applied)
        
        # Store AI recommendations in trainer for logging
        trainer.set_ai_recommendations(ai_log_data)
        logger.info("AI recommendations and reasoning stored for training log")
    
    def _display_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Display AI recommendations summary."""
        summary = create_advisor_summary(recommendations)
        logger.info(f"\n{summary}")
    
    def _save_recommendations(self, recommendations: Dict[str, Any]) -> None:
        """Save AI recommendations to file."""
        from trainio import setup_model_directory
        
        model_dir = setup_model_directory(self.config)
        rec_path = self.advisor.save_recommendations(
            recommendations,
            os.path.join(model_dir, 'ai_recommendations.json')
        )
        logger.info(f"AI recommendations saved to: {rec_path}")
    
    def _apply_recommendations(self, trainer, recommendations: Dict[str, Any], 
                             original_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AI recommendations and return applied changes."""
        logger.info("Applying AI recommendations to trainer configuration...")
        
        # Debug logging
        self._log_recommendation_structure(recommendations)
        
        if not self.advisor.apply_recommendations(trainer, recommendations):
            logger.warning("Failed to apply some AI recommendations")
            return {}
        
        logger.info("Successfully applied AI recommendations")
        
        # Track and log configuration changes
        changes_applied = self._track_config_changes(trainer, original_config)
        
        # Rebuild trainer components if necessary
        self._rebuild_trainer_if_needed(trainer, original_config)
        
        return changes_applied
    
    def _log_recommendation_structure(self, recommendations: Dict[str, Any]) -> None:
        """Log the structure of recommendations for debugging."""
        logger.info(f"Raw recommendations keys: {list(recommendations.keys())}")
        
        if "hyperparameters" not in recommendations:
            return
        
        hyperparams = recommendations["hyperparameters"]
        logger.info(f"Hyperparameters keys: {list(hyperparams.keys())}")
        
        if "training_config" in hyperparams:
            training_config = hyperparams["training_config"]
            logger.info(f"Training config keys: {list(training_config.keys())}")
    
    def _extract_reasoning_from_recommendations(self) -> Dict[str, str]:
        """Extract reasoning from the current AI recommendations."""
        if not hasattr(self, '_current_recommendations') or not self._current_recommendations:
            return {}
        
        reasoning_map = {}
        recommendations = self._current_recommendations
        
        # Extract reasoning from hyperparameters section
        if "hyperparameters" in recommendations:
            hyperparams = recommendations["hyperparameters"]
            
            # Training config reasoning
            if "training_config" in hyperparams:
                for param, details in hyperparams["training_config"].items():
                    if isinstance(details, dict) and "reasoning" in details:
                        reasoning_map[self._map_param_name(param)] = details["reasoning"]
            
            # Fine-tuning config reasoning
            if "fine_tuning_config" in hyperparams:
                for param, details in hyperparams["fine_tuning_config"].items():
                    if isinstance(details, dict) and "reasoning" in details:
                        reasoning_map[self._map_param_name(param)] = details["reasoning"]
            
            # Model architecture reasoning
            if "model_architecture" in hyperparams:
                arch_details = hyperparams["model_architecture"]
                if "base_model" in arch_details and isinstance(arch_details["base_model"], dict):
                    if "reasoning" in arch_details["base_model"]:
                        reasoning_map["base_model_name"] = arch_details["base_model"]["reasoning"]
        
        # Extract reasoning from optimization recommendations
        if "optimization_recommendations" in recommendations:
            opt_recs = recommendations["optimization_recommendations"]
            
            for section in ["training_config", "fine_tuning_config", "model_architecture"]:
                if section in opt_recs:
                    for param, details in opt_recs[section].items():
                        if isinstance(details, dict) and "reasoning" in details:
                            reasoning_map[self._map_param_name(param)] = details["reasoning"]
        
        return reasoning_map
    
    def _map_param_name(self, param_name: str) -> str:
        """Map AI recommendation parameter names to trainer attribute names."""
        param_mapping = {
            "batch_size": "batch_size",
            "initial_learning_rate": "initial_learning_rate",
            "fine_tune_learning_rate": "fine_tune_learning_rate",
            "initial_epochs": "initial_epochs",
            "fine_tune_epochs": "fine_tune_epochs",
            "unfreeze_percent": "unfreeze_percent",
            "image_size": "custom_img_size",
            "dual_stage": "dual_stage",
            "base_model": "base_model_name",
            "base_model_name": "base_model_name"
        }
        return param_mapping.get(param_name, param_name)
    
    def _track_config_changes(self, trainer, original_config: Dict[str, Any]) -> Dict[str, Any]:
        """Track and log configuration changes with reasoning."""
        updated_config = self.advisor.get_current_config(trainer)
        changes_applied = {}
        
        # Get reasoning from stored AI recommendations
        reasoning_map = self._extract_reasoning_from_recommendations()
        
        logger.info("=== AI CONFIGURATION CHANGES APPLIED ===")
        
        for key, new_value in updated_config.items():
            original_value = original_config.get(key)
            
            if original_value is None:
                # New configuration parameter
                reasoning = reasoning_map.get(key, "AI recommendation - no specific reasoning provided")
                logger.info(f"  ✓ {key}: {new_value} (new)")
                logger.info(f"    💡 Reasoning: {reasoning}")
                print(f"🤖 AI CHANGE: {key} = {new_value} (new)")
                print(f"   💡 Why: {reasoning}")
                changes_applied[key] = {
                    "original": None,
                    "applied": new_value,
                    "source": "ai_recommendation",
                    "reasoning": reasoning
                }
            elif original_value != new_value:
                # Changed configuration parameter
                reasoning = reasoning_map.get(key, "AI recommendation - no specific reasoning provided")
                logger.info(f"  ✓ {key}: {original_value} → {new_value}")
                logger.info(f"    💡 Reasoning: {reasoning}")
                print(f"🤖 AI CHANGE: {key} = {original_value} → {new_value}")
                print(f"   💡 Why: {reasoning}")
                changes_applied[key] = {
                    "original": original_value,
                    "applied": new_value,
                    "source": "ai_recommendation",
                    "reasoning": reasoning
                }
        
        if changes_applied:
            logger.info(f"=== {len(changes_applied)} AI CHANGES APPLIED ===")
            print(f"🎯 Applied {len(changes_applied)} AI-recommended changes to improve training performance")
        else:
            logger.info("=== NO AI CHANGES NEEDED ===")
            print("ℹ️  No configuration changes recommended by AI advisor")
        
        return changes_applied
    
    def _rebuild_trainer_if_needed(self, trainer, original_config: Dict[str, Any]) -> None:
        """Rebuild trainer components if image size or model changed."""
        updated_config = self.advisor.get_current_config(trainer)
        
        needs_rebuild = (
            original_config.get('img_size_used') != updated_config.get('img_size_used') or
            original_config.get('base_model_name') != updated_config.get('base_model_name')
        )
        
        if needs_rebuild:
            logger.info("Image size or model changed, rebuilding trainer components...")
            trainer.build_datasets()
            trainer.build()
if __name__ == "__main__":
    import json
    test = TrainingAdvisor()