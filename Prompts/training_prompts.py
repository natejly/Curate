"""
LLM Prompts for AI-Powered Training Optimization (Convergence-first)
These prompts are designed to drive reliable model convergence by:
1) prioritizing hyperparameter search,
2) only then enabling dual-stage (frozen → fine-tune),
3) only then considering a larger backbone.
"""

def get_initial_parameter_prompt(dataset_info, target_accuracy=99, debug = False):
    if debug:
        print(f"""Debugging initial parameter prompt generation...
              Task: {dataset_info['prompt']}\n
              Image Size: {dataset_info['img_size']}\n
              Number of Classes: {dataset_info['num_classes']}\n
              Classes: {dataset_info['classes']}\n
              File Tree Structure: {dataset_info['file_tree']}\n
              Dataset Splits: {dataset_info['dataset_splits']}\n
              """)
    """
    Prompt for initial parameter recommendation based on dataset analysis.
    Convergence-first: start simple; search hyperparameters before changing training stages or model size.
    
    Args:
        dataset_info: Dictionary containing dataset metadata
        target_accuracy: Target accuracy percentage (default: 99)
    
    Returns:
        String containing the formatted prompt
    """
    return f"""
You are selecting training parameters for an image classification run using ImgClassTrainer.

Dataset Info:
- Task: {dataset_info['prompt']}
- Image Size: {dataset_info['img_size']}
- Number of Classes: {dataset_info['num_classes']}
- Classes: {dataset_info['classes']}
- File Tree Structure: {dataset_info['file_tree']}
- Dataset Splits: {dataset_info['dataset_splits']}

GOAL:
- Recommend a single best-guess configuration that maximizes the chance of convergence toward ~{target_accuracy}% accuracy with minimal complexity.

SEARCH-FIRST STRATEGY (strict order):
1) HYPERPARAMETERS: Prefer single-stage training (dual_stage=False). Explore learning rate, batch size, and epochs first.
2) DUAL STAGE: Only if simple hyperparameter adjustments are unlikely to reach target, enable dual_stage=True (frozen backbone → fine-tune a subset of top layers).
3) LARGER MODEL: Only if (1) and (2) are insufficient, recommend a larger EfficientNet variant.

HARD RULES:
- If dual_stage is False, do NOT change any 2nd-stage parameters in spirit or in narrative. Concretely:
  • fine_tune_epochs must be 0
  • unfreeze_percent must be 0.0
  • fine_tune_learning_rate should equal initial_learning_rate (it will be ignored)
- If you set dual_stage to True, you may set fine_tune_epochs > 0, a smaller fine_tune_learning_rate than initial_learning_rate, and unfreeze_percent in [0.05–0.5] unless dataset is tiny.
- If specifying custom_img_size, both dimensions must be ≥ 64.

OUTPUT FORMAT:

EXPLANATION:
[Explain the recommendation concisely. START SIMPLE. Justify why hyperparameter choices are appropriate. If you escalate beyond single-stage, clearly justify why.]

PARAMETERS:
[Return ONLY a valid Python dictionary with these keys and types:]
- base_model_name (string: EfficientNetB0, EfficientNetB1, …)
- batch_size (int)
- initial_learning_rate (float)
- fine_tune_learning_rate (float)
- initial_epochs (int)
- fine_tune_epochs (int)  # must be 0 if dual_stage=False
- dual_stage (bool)       # prefer False initially
- custom_img_size (list like [height, width] with min 64x64, or None)
- unfreeze_percent (float 0.0–1.0)  # must be 0.0 if dual_stage=False

DEFAULT STARTING POINT:
- Favor EfficientNetB0, dual_stage=False, simple single-stage schedule.

IMPORTANT SYNTAX:
- Use Python booleans (True/False), None (not null), and numeric literals (no quotes).
"""


def get_feedback_prompt(final_accuracy, target_accuracy, training_log, current_params, user_task="", dataset_info=None, debug=False):
    """
    Structured prompt for iterative tuning with strict escalation:
    1) Hyperparameter search → 2) Dual stage → 3) Larger model.
    Also enforces: if dual_stage=False, do not alter any 2nd-stage parameters.
    
    Args:
        final_accuracy (float): Achieved accuracy
        target_accuracy (float): Desired target accuracy (e.g., 0.99 for 99%)
        training_log (str): Training log data
        current_params (dict): Current parameter configuration
        user_task (str): Original task description
        dataset_info (dict): Optional dataset metadata
    
    Returns:
        str: Formatted prompt string
    """
    if debug:
        print(f"""Debugging initial parameter prompt generation...
              Task: {dataset_info['prompt']}
              Image Size: {dataset_info['img_size']}
              Number of Classes: {dataset_info['num_classes']}
              Classes: {dataset_info['classes']}
              File Tree Structure: {dataset_info['file_tree']}
              Dataset Splits: {dataset_info['dataset_splits']}
              """)

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
Training Feedback Request (Convergence-first)
============================================

Original Task: {user_task}
Target Accuracy: {target_accuracy*100:.2f}%
{dataset_section}
Observed Results:
- Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)
- Training Log: {training_log}
- Current Parameters: {current_params}

YOUR JOB:
Diagnose why we missed target and recommend the next configuration that most likely improves accuracy with the least complexity.

STRICT ESCALATION ORDER:
1) Hyperparameter adjustments (learning rate(s), batch size, epochs, optional custom_img_size) while keeping dual_stage=False.
2) If (1) is insufficient or already tried with reasonable coverage, enable dual_stage=True and specify fine-tuning details (smaller LR, unfreeze top fraction, finite fine_tune_epochs).
3) If (2) appears insufficient, suggest a larger EfficientNet variant (e.g., B1/B2/…).

CRITICAL CONSTRAINTS:
- If dual_stage is False:
  • Do NOT propose or imply any change to 2nd-stage behavior.
  • Set fine_tune_epochs=0, unfreeze_percent=0.0, fine_tune_learning_rate=initial_learning_rate.
- Only modify fine_tune_epochs and unfreeze_percent when dual_stage=True.
- custom_img_size, if used, must have both dimensions ≥ 64.
- Keep changes minimal and targeted; avoid changing many variables at once unless justified by logs.

RESPONSE FORMAT
---------------
EXPLANATION:
[Short diagnosis referencing the logs and results. State clearly which escalation level you’re using and why. If staying in level 1, explain the hyperparameter search direction (e.g., LR decay or warmup tweaks, modest batch changes, epochs).]

PARAMETERS:
[Return ONLY a valid Python dictionary with exactly these keys:]
- base_model_name (EfficientNetB0, EfficientNetB1, etc.)
- batch_size (int)
- initial_learning_rate (float)
- fine_tune_learning_rate (float)
- initial_epochs (int)
- fine_tune_epochs (int)          # must be 0 if dual_stage=False
- dual_stage (bool)
- custom_img_size (list like [h, w] ≥ [64, 64], or None)
- unfreeze_percent (float 0.0–1.0) # must be 0.0 if dual_stage=False

SYNTAX:
- Proper Python: True/False, None, decimals for floats, no quotes on numbers.
- Do NOT add keys beyond the list above.
"""


def get_parameter_descriptions():
    """
    Returns descriptions of all available parameters for reference,
    annotated with convergence-first guidance.
    
    Returns:
        Dictionary containing parameter descriptions
    """
    return {
        "base_model_name": "EfficientNet variant (B0–B7). Start with B0; escalate only after HP search and (optionally) dual-stage fail to converge.",
        "batch_size": "Training batch size. Increase for stability/throughput if memory allows; decrease if overfitting or OOM. Tune before changing model size.",
        "initial_learning_rate": "Stage-1 LR (frozen backbone in dual-stage; full training if single-stage). Primary lever for convergence — explore first.",
        "fine_tune_learning_rate": "Stage-2 LR (used only when dual_stage=True). Typically lower than initial_learning_rate.",
        "initial_epochs": "Epochs for stage 1. Adjust moderately during HP search to balance under/overfitting before escalating.",
        "fine_tune_epochs": "Epochs for stage 2 (only when dual_stage=True). Must be 0 when dual_stage=False.",
        "dual_stage": "False for single-stage (default and preferred initially). Switch to True only after reasonable HP search fails.",
        "custom_img_size": "Optional [height, width] (each ≥ 64). Use to reduce compute or fit memory; keep None unless clearly beneficial.",
        "unfreeze_percent": "Fraction (0.0–1.0) of top layers to unfreeze in stage 2. Must be 0.0 when dual_stage=False."
    }
