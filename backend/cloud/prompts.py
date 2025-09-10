"""
Simplified AI Advisor prompts that rely on RAG for context.
Knowledge is retrieved from Pinecone and injected into prompts.
"""

# Simplified system prompt that relies on RAG context
HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT = """
You are an expert machine learning engineer specializing in computer vision and transfer learning optimization.

Your task is to analyze dataset characteristics and recommend optimal hyperparameters for training image classification models.

You will be provided with relevant knowledge context to inform your recommendations.

You must provide responses in valid JSON format only, with detailed reasoning for each recommendation.
"""

# Simplified hyperparameter request template
HYPERPARAMETER_REQUEST_TEMPLATE = """
Analyze the following dataset characteristics and provide optimized hyperparameters for image classification training:

Dataset Information:
{dataset_info}

Current Configuration:
{current_config}

Requirements:
1. Recommend optimal hyperparameters for both single-stage and dual-stage training
2. Consider the dataset size, image dimensions, and class distribution
3. Provide reasoning for each recommendation
4. Choose between single-stage or dual-stage training approach

IMPORTANT JSON FORMAT REQUIREMENTS:
1. Use exactly the key "value" (not "recommended_value" or any other variant) for all parameter values.
2. Return values in their proper data types:
   - batch_size: integer (e.g., 64, not "64")
   - learning rates: float (e.g., 0.001, not "0.001")
   - epochs: integer (e.g., 20, not "20")
   - image_size: array of two integers (e.g., [224, 224], not ["224", "224"])
   - unfreeze_percent: float between 0 and 1 (e.g., 0.5, not "0.5")

Respond with the following JSON structure exactly as shown:
{{
  "analysis": {{
    "dataset_complexity": "low|medium|high",
    "recommended_approach": "single_stage|dual_stage",
    "key_insights": ["insight1", "insight2", "..."]
  }},
  "hyperparameters": {{
    "model_architecture": {{
      "base_model": {{
        "value": "string_model_name",
        "reasoning": "explanation"
      }}
    }},
    "training_config": {{
      "batch_size": {{
        "value": "integer_value_based_on_dataset",
        "reasoning": "explanation"
      }},
      "initial_learning_rate": {{
        "value": "float_value_based_on_model",
        "reasoning": "explanation"
      }},
      "initial_epochs": {{
        "value": "integer_value_based_on_dataset",
        "reasoning": "explanation"
      }},
      "image_size": {{
        "value": ["integer_width", "integer_height"],
        "reasoning": "explanation"
      }},
      "dual_stage": {{
        "value": "boolean_true_or_false",
        "reasoning": "explanation"
      }}
    }},
    "fine_tuning_config": {{
      "fine_tune_learning_rate": {{
        "value": "float_value_for_fine_tuning",
        "reasoning": "explanation"
      }},
      "fine_tune_epochs": {{
        "value": "integer_value_for_convergence",
        "reasoning": "explanation"
      }},
      "unfreeze_percent": {{
        "value": "float_0_to_1_for_layers",
        "reasoning": "explanation"
      }}
    }}
  }}
}}
"""

# Simplified optimization system prompt
OPTIMIZATION_SYSTEM_PROMPT = """You are an expert machine learning engineer specializing in training optimization. 

Analyze training logs and make SUBSTANTIAL changes that will have significant impact on model performance.

You will be provided with relevant knowledge context to guide your optimization decisions.

Your goal is to make hyperparameter adjustments that will meaningfully improve training performance.

Respond with valid JSON only using the specified format.
"""

# Simplified optimization request template
OPTIMIZATION_REQUEST_TEMPLATE = """
Analyze training results and make SUBSTANTIAL optimizations that will significantly impact performance:

CURRENT CONFIG:
{current_config}

TRAINING DATA:
{training_log}

REQUIREMENTS:
- Focus on hyperparameter optimization first
- Make substantial changes that will create measurable performance differences
- Avoid minor tweaks - recommend bold adjustments
- Provide clear reasoning for each optimization
- Only recommend architecture changes if absolutely necessary with strong justification

RESPONSE FORMAT - Valid JSON only:
{{
    "analysis": {{
        "performance_assessment": "brief assessment",
        "identified_issues": ["specific issues"],
        "recommended_approach": "single_stage|dual_stage"
    }},
    "optimization_recommendations": {{
        "training_config": {{
            "batch_size": {{
                "recommended_value": "integer_value",
                "reasoning": "explanation"
            }},
            "initial_learning_rate": {{
                "recommended_value": "float_value",
                "reasoning": "explanation"
            }},
            "initial_epochs": {{
                "recommended_value": "integer_value",
                "reasoning": "explanation"
            }},
            "image_size": {{
                "recommended_value": ["width", "height"],
                "reasoning": "explanation"
            }},
            "dual_stage": {{
                "recommended_value": "boolean_value",
                "reasoning": "explanation"
            }}
        }},
        "fine_tuning_config": {{
            "fine_tune_learning_rate": {{
                "recommended_value": "float_value",
                "reasoning": "explanation"
            }},
            "fine_tune_epochs": {{
                "recommended_value": "integer_value",
                "reasoning": "explanation"
            }},
            "unfreeze_percent": {{
                "recommended_value": "float_value",
                "reasoning": "explanation"
            }}
        }},
        "model_architecture": {{
            "base_model_name": {{
                "recommended_value": "string_model_name",
                "reasoning": "explanation - ONLY recommend if hyperparameters cannot solve fundamental capacity issues"
            }}
        }}
    }}
}}

PARAMETER TYPES:
- batch_size: integer (power of 2 values)
- learning_rates: float (scientific notation format)  
- epochs: integer (positive values)
- image_size: [width, height] integers (square or rectangular)
- unfreeze_percent: float 0.0-1.0 (percentage as decimal)
- dual_stage: boolean (true for dual-stage, false for single-stage)
- base_model_name: string (EfficientNet family)
"""

def get_hyperparameter_prompt(dataset_info, current_config):
    """Generate the complete prompt for hyperparameter optimization."""
    return HYPERPARAMETER_REQUEST_TEMPLATE.format(
        dataset_info=dataset_info,
        current_config=current_config
    )

def get_dataset_analysis_prompt(dataset_details):
    """Generate basic prompt for dataset analysis focused on hyperparameter selection."""
    return f"""Analyze this dataset for hyperparameter optimization:

Dataset Details:
{dataset_details}

Focus on aspects that affect hyperparameter selection:
1. Dataset size and its impact on batch size and learning rate
2. Image dimensions and their impact on optimal input size
3. Class distribution and its impact on training approach
4. Dataset complexity and its impact on model architecture choice

Provide insights that will help optimize hyperparameters, image size, and model architecture."""

def get_optimization_prompt(training_log, current_config):
    """Generate prompt for training optimization."""
    return OPTIMIZATION_REQUEST_TEMPLATE.format(
        training_log=training_log,
        current_config=current_config
    )
