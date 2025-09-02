"""
AI Advisor prompts for hyperparameter optimization
Contains system prompts and templates for OpenAI API calls
"""

HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT = """
You are an expert machine learning engineer specializing in computer vision and transfer learning optimization.

Your task is to analyze dataset characteristics and recommend optimal hyperparameters for training image classification models on AWS SageMaker using TensorFlow.

Key areas of expertise:
- Transfer learning with pre-trained models (EfficientNet, ResNet, Vision Transformers, etc.)
- Learning rate scheduling and optimization strategies
- Batch size optimization based on dataset size and complexity
- Training duration and epoch recommendations
- Data augmentation strategies
- Model architecture selection

You must provide responses in valid JSON format only, with detailed reasoning for each recommendation.
"""

HYPERPARAMETER_REQUEST_TEMPLATE = """
Analyze the following dataset characteristics and provide optimized hyperparameters for image classification training:

Dataset Information:
{dataset_info}

Current Configuration:
{current_config}

Requirements:
1. Recommend optimal hyperparameters for both single-stage and dual-stage training
2. Consider the dataset size, image dimensions, and class distribution
3. Optimize for AWS SageMaker constraints and best practices
4. Provide reasoning for each recommendation
5. Include confidence scores (0-100) for each recommendation

IMPORTANT:
1. Use exactly the key "value" (not "recommended_value" or any other variant) for all parameter values.
2. Return values in their proper data types:
   - batch_size: integer (e.g., 64, not "64")
   - learning rates: float (e.g., 0.001, not "0.001")
   - epochs: integer (e.g., 20, not "20")
   - image_size: array of two integers (e.g., [224, 224], not ["224", "224"] or "[224, 224]")
   - unfreeze_percent: float between 0 and 1 (e.g., 0.5, not "0.5")
   - confidence: integer between 0-100 (e.g., 85, not "85")

Respond with the following JSON structure exactly as shown:
{{
  "analysis": {{
    "dataset_complexity": "low|medium|high",
    "recommended_approach": "single_stage|dual_stage",
    "key_insights": ["insight1", "insight2", "..."],
    "potential_challenges": ["challenge1", "challenge2", "..."]
  }},
  "hyperparameters": {{
    "model_architecture": {{
      "base_model": {{
        "value": "string_model_name",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }}
    }},
    "training_config": {{
      "batch_size": {{
        "value": "integer_value_based_on_dataset",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }},
      "initial_learning_rate": {{
        "value": "float_value_based_on_model",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }},
      "initial_epochs": {{
        "value": "integer_value_based_on_dataset",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }},
      "image_size": {{
        "value": ["integer_width", "integer_height"],
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }}
    }},
    "fine_tuning_config": {{
      "fine_tune_learning_rate": {{
        "value": "float_value_for_fine_tuning",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }},
      "fine_tune_epochs": {{
        "value": "integer_value_for_convergence",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }},
      "unfreeze_percent": {{
        "value": "float_0_to_1_for_layers",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }}
    }},
    "optimization": {{
      "optimizer": {{
        "type": "string_optimizer_name",
        "weight_decay": "float_regularization_value",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }},
      "scheduler": {{
        "type": "string_scheduler_name",
        "factor": "float_decay_factor",
        "patience": "integer_patience_value",
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }}
    }}
  }},
  "sagemaker_recommendations": {{
    "instance_type": {{
      "value": "string_instance_type",
      "confidence": "integer_0_to_100",
      "reasoning": "explanation"
    }},
    "estimated_training_time": {{
      "single_stage": "string_time_estimate",
      "dual_stage": "string_time_estimate"
    }},
    "cost_estimate": {{
      "approximate_cost": "string_cost_range",
      "reasoning": "explanation"
    }}
  }},
  "data_augmentation": {{
    "recommended_augmentations": [
      {{
        "type": "string_augmentation_type",
        "parameters": {{"string_param_name": "appropriate_param_value"}},
        "confidence": "integer_0_to_100",
        "reasoning": "explanation"
      }}
    ]
  }}
}}
"""

DATASET_ANALYSIS_PROMPT = """
Based on the dataset characteristics provided, analyze the data distribution and provide insights:

Dataset Details:
{dataset_details}

Focus on:
1. Class balance and potential bias issues
2. Image quality and consistency
3. Dataset size adequacy for transfer learning
4. Potential overfitting risks
5. Recommended validation split strategy

Provide analysis in JSON format with actionable recommendations.

Expected JSON structure:
{{
  "dataset_analysis": {{
    "class_balance": {{
      "status": "balanced|imbalanced|severely_imbalanced",
      "recommendations": ["recommendation1", "recommendation2"],
      "confidence": "<0-100>"
    }},
    "image_quality": {{
      "assessment": "high|medium|low|mixed",
      "issues": ["issue1", "issue2"],
      "recommendations": ["recommendation1", "recommendation2"]
    }},
    "dataset_adequacy": {{
      "size_assessment": "sufficient|marginal|insufficient",
      "transfer_learning_viability": "high|medium|low",
      "recommendations": ["recommendation1", "recommendation2"]
    }},
    "overfitting_risk": {{
      "risk_level": "low|medium|high",
      "mitigation_strategies": ["strategy1", "strategy2"]
    }},
    "validation_strategy": {{
      "recommended_split": {{
        "train_percent": "<percentage>",
        "validation_percent": "<percentage>",
        "test_percent": "<percentage>"
      }},
      "split_method": "random|stratified|temporal|custom",
      "reasoning": "explanation"
    }}
  }}
}}
"""

ERROR_ANALYSIS_PROMPT = """
Analyze the training results and suggest improvements:

Training History:
{training_history}

Current Performance:
{performance_metrics}

Previous Hyperparameters:
{previous_config}

Identify issues and suggest hyperparameter adjustments to improve performance.

Expected JSON response structure:
{{
  "performance_analysis": {{
    "identified_issues": [
      {{
        "issue": "overfitting|underfitting|poor_convergence|class_imbalance|etc",
        "evidence": "specific evidence from metrics",
        "severity": "low|medium|high"
      }}
    ],
    "root_causes": ["cause1", "cause2"],
    "improvement_potential": "low|medium|high"
  }},
  "hyperparameter_adjustments": {{
    "learning_rate": {{
      "current_value": "<current>",
      "suggested_value": "<suggested>",
      "reasoning": "explanation",
      "priority": "low|medium|high"
    }},
    "batch_size": {{
      "current_value": "<current>",
      "suggested_value": "<suggested>",
      "reasoning": "explanation",
      "priority": "low|medium|high"
    }},
    "model_architecture": {{
      "current_value": "<current>",
      "suggested_value": "<suggested>",
      "reasoning": "explanation",
      "priority": "low|medium|high"
    }},
    "regularization": {{
      "current_techniques": ["technique1", "technique2"],
      "suggested_techniques": ["technique1", "technique2"],
      "reasoning": "explanation",
      "priority": "low|medium|high"
    }},
    "data_augmentation": {{
      "current_augmentations": ["aug1", "aug2"],
      "suggested_augmentations": ["aug1", "aug2"],
      "reasoning": "explanation",
      "priority": "low|medium|high"
    }}
  }},
  "training_strategy": {{
    "recommended_approach": "continue_training|restart_with_changes|multi_stage_training",
    "next_steps": ["step1", "step2", "step3"],
    "expected_improvement": "estimated improvement description"
  }}
}}
"""

def get_hyperparameter_prompt(dataset_info, current_config):
    """Generate the complete prompt for hyperparameter optimization."""
    return HYPERPARAMETER_REQUEST_TEMPLATE.format(
        dataset_info=dataset_info,
        current_config=current_config
    )

def get_dataset_analysis_prompt(dataset_details):
    """Generate prompt for dataset analysis."""
    return DATASET_ANALYSIS_PROMPT.format(dataset_details=dataset_details)

def get_error_analysis_prompt(training_history, performance_metrics, previous_config):
    """Generate prompt for training error analysis."""
    return ERROR_ANALYSIS_PROMPT.format(
        training_history=training_history,
        performance_metrics=performance_metrics,
        previous_config=previous_config
    )