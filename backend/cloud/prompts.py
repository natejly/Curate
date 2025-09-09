"""
Streamlined AI Advisor prompts for hyperparameter optimization.
Focused on hyperparameters, image size, and model architecture only.
"""

HYPERPARAMETER_ADVISOR_SYSTEM_PROMPT = """
You are an expert machine learning engineer specializing in computer vision and transfer learning optimization.

Your task is to analyze dataset characteristics and recommend optimal hyperparameters for training image classification models.

Key areas of expertise:
- Transfer learning with pre-trained models (EfficientNet)
- Learning rate optimization for initial training and fine-tuning
- Batch size optimization based on dataset size and complexity
- Training duration and epoch recommendations
- Image size selection for optimal performance
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
3. Provide reasoning for each recommendation
4. Choose between single-stage or dual-stage training approach

TRAINING APPROACHES EXPLAINED:
- **Single-stage training**: Trains the entire model from the start with unfrozen layers. Good for larger datasets where you want immediate feature adaptation.
- **Dual-stage training**: First trains only the top layers (feature extractor frozen), then unfreezes a percentage of layers for fine-tuning. Better for smaller datasets or when transfer learning benefits are important.

IMPORTANT:
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

OPTIMIZATION_SYSTEM_PROMPT = """You are an expert machine learning engineer specializing in training optimization. 
Analyze training logs and make SUBSTANTIAL changes that will have significant impact on model performance.

ESCALATION ORDER - Apply changes in this priority:
1. **HYPERPARAMETERS** (Primary): learning_rates, batch_size, epochs, unfreeze_percent, image_size
3. **ARCHITECTURE** (Last resort): Change base_model when hyperparameters aren't sufficient

MAKE BOLD CHANGES:
- Don't suggest minor tweaks - recommend changes that will meaningfully impact training
- If learning rate is causing issues, make significant adjustments (order of magnitude changes, not small increments)
- If batch size is suboptimal, recommend substantial changes based on dataset size and memory constraints
- If model is struggling, consider jumping to significantly larger or smaller architectures

TRAINING APPROACHES:
- Single-stage (dual_stage=false): Trains entire model with unfrozen layers. Good for larger datasets.
- Dual-stage (dual_stage=true): First trains top layers, then fine-tunes. Better for smaller datasets.

RESPONSE FORMAT - Valid JSON only:
{
    "analysis": {
        "performance_assessment": "brief assessment",
        "identified_issues": ["specific issues"],
        "recommended_approach": "single_stage|dual_stage"
    },
    "optimization_recommendations": {
        "training_config": {
            "batch_size": {
                "recommended_value": "integer_value",
                "reasoning": "explanation"
            },
            "initial_learning_rate": {
                "recommended_value": "float_value",
                "reasoning": "explanation"
            },
            "initial_epochs": {
                "recommended_value": "integer_value",
                "reasoning": "explanation"
            },
            "image_size": {
                "recommended_value": ["width", "height"],
                "reasoning": "explanation"
            }
        },
        "fine_tuning_config": {
            "fine_tune_learning_rate": {
                "recommended_value": "float_value",
                "reasoning": "explanation"
            },
            "fine_tune_epochs": {
                "recommended_value": "integer_value",
                "reasoning": "explanation"
            },
            "unfreeze_percent": {
                "recommended_value": "float_value",
                "reasoning": "explanation"
            }
        },
        "model_architecture": {
            "base_model_name": {
                "recommended_value": "string_model_name",
                "reasoning": "explanation"
            }
        }
    }
}

PARAMETER TYPES:
- batch_size: integer (power of 2 values)
- learning_rates: float (scientific notation format)  
- epochs: integer (positive values)
- image_size: [width, height] integers (square or rectangular)
- unfreeze_percent: float 0.0-1.0 (percentage as decimal)
- base_model_name: string (EfficientNet family)

Only recommend changes needed for optimization."""

OPTIMIZATION_REQUEST_TEMPLATE = """
Analyze training results and make SUBSTANTIAL optimizations that will significantly impact performance:

CURRENT CONFIG:
{current_config}

TRAINING DATA:
{training_log}

ESCALATION ANALYSIS (in priority order):
1. **HYPERPARAMETERS FIRST**: 
   - Learning rates: If problematic, make order-of-magnitude adjustments
   - Batch size: Make significant changes based on dataset size and memory constraints
   - Epochs: Substantial increases/decreases if under/overfitting detected
   - Unfreeze percent: Bold adjustments if fine-tuning issues identified
   - Image size: Optimize input dimensions for model and dataset

2. **ARCHITECTURE LAST**: 
   - Only if hyperparameters insufficient
   - Jump between model sizes for substantial capacity changes (smaller for overfitting, larger for underfitting)

IMPACT REQUIREMENTS:
- Make changes that will create measurable performance differences
- Avoid minor tweaks - recommend bold adjustments
- Prioritize changes with highest expected impact
- Explain why each change will substantially improve training

GOALS:
- Achieve significant validation accuracy improvements
- Dramatically improve training efficiency
- Make substantial impact on model performance

Focus on high-impact changes with clear reasoning for substantial improvements.
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