"""
LLM Prompts for AI-Powered Training Optimization
This file contains all the prompts used for GPT-4o integration in the training system.
"""

def get_initial_parameter_prompt(dataset_info, target_accuracy=99):
    """
    Prompt for initial parameter recommendation based on dataset analysis.
    
    Args:
        dataset_info: Dictionary containing dataset metadata
        target_accuracy: Target accuracy percentage (default: 99)
    
    Returns:
        String containing the formatted prompt
    """
    return f"""
Given this image classification dataset analysis, recommend optimal training parameters for ImgClassTrainer:

Dataset Info:
- Task: {dataset_info['prompt']}
- Image Size: {dataset_info['img_size']}
- Number of Classes: {dataset_info['num_classes']}
- Classes: {dataset_info['classes']}
- File Tree Structure: {dataset_info['file_tree']}
- Dataset Splits: {dataset_info['dataset_splits']}

Please recommend parameters for:
1. base_model_name (EfficientNetB0, EfficientNetB1, etc.)
2. batch_size
3. initial_learning_rate
4. fine_tune_learning_rate
5. initial_epochs
6. fine_tune_epochs
7. dual_stage (True for two-stage training, False for single-stage). 
8. custom_img_size (tuple like (height, width) or null to use auto-bucketing)
9. unfreeze_percent (float 0.0-1.0, percentage of top layers to unfreeze in stage 2)

Please provide your response in the following format:

EXPLANATION:
[Explain your parameter choices based on the dataset characteristics, task complexity, and optimal training strategy. START WITH SIMPLE SOLUTIONS FIRST.]

PARAMETERS:
[Return ONLY a valid Python dictionary with these parameters:]
- base_model_name (string: EfficientNetB0, EfficientNetB1, etc.)
- batch_size (integer)
- initial_learning_rate (float)
- fine_tune_learning_rate (float)
- initial_epochs (integer)
- fine_tune_epochs (integer)
- dual_stage (boolean: False for single-stage, True for two-stage - prefer False initially)
- custom_img_size (list: [height, width] with minimum 64x64, or None for auto-bucketing)
- unfreeze_percent (float: 0.0-1.0, percentage of top layers to unfreeze)

STRATEGY: Prioritize simple solutions first - start with single-stage training (dual_stage=False) and EfficientNetB0.
CONSTRAINTS: If specifying custom_img_size, ensure both height and width are at least 64 pixels.
IMPORTANT: Use Python syntax (True/False not true/false, None not null, numbers without quotes)

Consider the dataset size, image dimensions, and task complexity.
"""


def get_feedback_prompt(final_accuracy, target_accuracy, training_log, current_params, user_task="", dataset_info=None):
    """
    Generate a structured prompt for iterative parameter tuning based on training results.
    
    Args:
        final_accuracy (float): Achieved accuracy
        target_accuracy (float): Desired target accuracy
        training_log (str): Training log data
        current_params (dict): Current parameter configuration
        user_task (str): Original user task description for context
        dataset_info (dict): Dataset metadata including structure and characteristics
    
    Returns:
        str: A formatted prompt string
    """
    dataset_section = ""
    if dataset_info:
        dataset_section = f"""
Dataset Context:
- Task: {dataset_info.get('prompt', 'N/A')}
- Image Size: {dataset_info.get('img_size', 'N/A')}
- Number of Classes: {dataset_info.get('num_classes', 'N/A')}
- Classes: {dataset_info.get('classes', 'N/A')}
- File Tree Structure: {dataset_info.get('file_tree', 'N/A')}
- Dataset Splits: {dataset_info.get('dataset_splits', 'N/A')}
"""
    
    return f"""
Training Feedback Request
=========================

Original Task: {user_task}
{dataset_section}
We attempted training with the following setup. Please analyze the results and suggest improvements to reach the target accuracy of {target_accuracy*100:.2f}%:

Current Results:
- Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)
- Training Log: {training_log}
- Current Parameters: {current_params}

Your task:
1. Diagnose possible reasons why accuracy did not reach the target.
2. Recommend specific parameter changes most likely to improve accuracy.
3. PRIORITIZE SIMPLE SOLUTIONS FIRST: Try hyperparameter adjustments before dual-stage training or larger models.

Response Format
---------------
EXPLANATION:
[Explain observations from the results, issues identified, and reasoning for the recommended changes. Focus on simple solutions first.]

PARAMETERS:
[Return ONLY a valid Python dictionary with exactly these keys:]
- base_model_name (EfficientNetB0, EfficientNetB1, etc.)
- batch_size (int)
- initial_learning_rate (float)
- fine_tune_learning_rate (float)
- initial_epochs (int)
- fine_tune_epochs (int)
- dual_stage (bool - only set to True if simple hyperparameter changes haven't worked)
- custom_img_size (list/tuple like [height, width] with minimum 64x64, or None)
- unfreeze_percent (float, range 0.0–1.0)

RULES:
- Use proper Python syntax (True/False, None, floats as decimals).
- Do NOT include keys like img_size, dataset_path, or any others.
- ONLY change fine_tune_epochs and unfreeze_percent if dual_stage is True
- ESCALATION ORDER: 1) Hyperparameters → 2) Dual-stage training → 3) Larger architecture
- IMAGE SIZE CONSTRAINT: If specifying custom_img_size, ensure both dimensions are at least 64 pixels.
- Focus only on parameter changes most likely to improve accuracy given the training history.
"""



def get_parameter_descriptions():
    """
    Returns descriptions of all available parameters for reference.
    
    Returns:
        Dictionary containing parameter descriptions
    """
    return {
        "base_model_name": "EfficientNet variant (B0-B7) - larger models for complex tasks",
        "batch_size": "Training batch size - smaller for limited memory, larger for stability", 
        "initial_learning_rate": "Learning rate for stage 1 (frozen backbone)",
        "fine_tune_learning_rate": "Learning rate for stage 2 (fine-tuning) - usually lower",
        "initial_epochs": "Epochs for stage 1 training",
        "fine_tune_epochs": "Epochs for stage 2 fine-tuning",
        "dual_stage": "True for two-stage training (frozen→fine-tune), False for single-stage",
        "custom_img_size": "Custom image dimensions [height, width] or None for auto-bucketing",
        "unfreeze_percent": "Percentage (0.0-1.0) of top layers to unfreeze in stage 2"
    }
