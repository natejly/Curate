import os
import openai
import time
import json
import re
from dotenv import load_dotenv
from ImgClassTrain import ImgClassTrainer
from ImgClassData import ImgClassData

# Load environment variables from .env file
load_dotenv()


# User prompt for the training task
prompt = "I want to train a digits classifier"

# Dataset path
dataset_path = "/Users/natejly/Desktop/sorted_digits"

# Initialize trainer variable
trainer_single = None

# Analyze the dataset first to get metadata
print("📊 Analyzing dataset...")
data_parser = ImgClassData(dataset_path, debug=False)

# Prepare data for LLM query
dataset_info = {
    "prompt": prompt,
    "img_size": data_parser.IMSIZE,
    "file_tree": json.loads(data_parser.json_tree),
    "dataset_splits": json.loads(data_parser.DS_split_tree),
    "classes": data_parser.classes,
    "num_classes": len(data_parser.classes)
}

print("✅ Dataset Analysis Complete:")
print(f"   Task: {prompt}")
print(f"   Image Size: {data_parser.IMSIZE}")
print(f"   Classes: {data_parser.classes}")
print(f"   Number of Classes: {len(data_parser.classes)}")

# Create query for GPT-4o
llm_query = f"""
Given this image classification dataset analysis, recommend optimal training parameters for ImgClassTrainer:

Dataset Info:
- Task: {dataset_info['prompt']}
- Image Size: {dataset_info['img_size']}
- Number of Classes: {dataset_info['num_classes']}
- Classes: {dataset_info['classes']}
- File Tree Structure: {json.dumps(dataset_info['file_tree'], indent=2)}
- Dataset Splits: {json.dumps(dataset_info['dataset_splits'], indent=2)}

Please recommend parameters for:
1. base_model_name (EfficientNetB0, EfficientNetB1, etc.)
2. batch_size
3. initial_learning_rate
4. fine_tune_learning_rate
5. initial_epochs
6. fine_tune_epochs
7. dual_stage 
8. custom_img_size 
9. unfreeze_percent

Please provide your response in the following format:

EXPLANATION:
[Explain your parameter choices based on the dataset characteristics, task complexity, and optimal training strategy.]

PARAMETERS:
[Return ONLY a valid Python dictionary with these parameters:]
- base_model_name (string: EfficientNetB0, EfficientNetB1, etc.)
- batch_size (integer)
- initial_learning_rate (float)
- fine_tune_learning_rate (float)
- initial_epochs (integer)
- fine_tune_epochs (integer)
- dual_stage (boolean: True for two-stage, False for single-stage)
- custom_img_size (list: [height, width] or None for auto-bucketing)
- unfreeze_percent (float: 0.0-1.0, percentage of top layers to unfreeze)

IMPORTANT: Use Python syntax (True/False not true/false, None not null, numbers without quotes)


Consider the dataset size, image dimensions, and task complexity.
"""

# Get OpenAI API key from environment
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    try:
        print("\n🤖 Querying GPT-4o for initial optimal parameters...")
        # Query GPT-4o for training parameters
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": llm_query}],
            temperature=0.1
        )
        
        llm_response = response.choices[0].message.content
        print("✅ Initial LLM Response received")
        
        # Extract explanation first
        explanation_match = re.search(r'EXPLANATION:\s*(.*?)\s*PARAMETERS:', llm_response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            print(f"\n💡 LLM Initial Parameter Reasoning:")
            print("="*60)
            print(explanation)
            print("="*60)
        else:
            print("\n⚠️ No explanation found in initial LLM response")
        
        # Extract dictionary from LLM response
        dict_match = re.search(r'\{[^}]*\}', llm_response, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group()
            dict_str = dict_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            recommended_params = eval(dict_str)
            
            print("\n📋 AI-Recommended Initial Parameters:")
            print(json.dumps(recommended_params, indent=2))
            
            # Create trainer with LLM-recommended parameters
            trainer_single = ImgClassTrainer(
                dataset_path=dataset_path,
                **recommended_params
            )
            print(f"\n✅ Initial trainer created with AI recommendations!")
            print(f"   Training mode: {trainer_single.get_training_mode()}")
        else:
            raise ValueError("Could not extract parameters from LLM response")
            
    except Exception as e:
        print(f"❌ Error with initial LLM query: {e}")
        trainer_single = None

if trainer_single is None:
    print("\n⚙️ Creating trainer with default parameters...")
    trainer_single = ImgClassTrainer(
        dataset_path=dataset_path,
        base_model_name="EfficientNetB0",
        batch_size=32,
        initial_learning_rate=1e-3,
        initial_epochs=2,
        dual_stage=False,
        custom_img_size=(224, 224),
        unfreeze_percent=0.3
    )
    print("✅ Trainer created with defaults")

print(f"\n🎯 Initial Configuration Ready:")
print(json.dumps(trainer_single.getParams(), indent=2))

max_iterations = 5
target_accuracy = 0.99
iteration = 1

print("🔄 Starting iterative LLM-guided training...")
print(f"Target accuracy: {target_accuracy*100}%")
print(f"Max iterations: {max_iterations}")

while iteration <= max_iterations:
    print(f"\n{'='*60}")
    print(f"🚀 ITERATION {iteration}/{max_iterations}")
    print(f"{'='*60}")
    
    # Show current parameters
    print(f"📊 Current Parameters:")
    print(json.dumps(trainer_single.getParams(), indent=2))
    
    # Run training
    print(f"\n🏋️ Training with current parameters...")
    trainer_single.run()
    
    # Get final test accuracy - handle different possible metric keys
    metrics = trainer_single.getMetrics()
    print(f"Debug - Raw metrics: {metrics}")
    
    # Try different possible keys for accuracy
    final_accuracy = 0.0
    if metrics:
        for key in ['accuracy', 'compile_metrics', 'val_accuracy']:
            if key in metrics:
                final_accuracy = metrics[key]
                print(f"Using accuracy from key: {key}")
                break
    
    print(f"\n📈 Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Check if target accuracy reached
    if final_accuracy >= target_accuracy:
        print(f"🎉 SUCCESS! Target accuracy of {target_accuracy*100}% reached!")
        print(f"✅ Training completed in {iteration} iteration(s)")
        break
    
    # If not at target and not last iteration, get LLM feedback
    if iteration < max_iterations:
        print(f"📊 Accuracy below target ({target_accuracy*100}%). Getting LLM feedback...")
        
        # Prepare training log for LLM analysis
        training_log = trainer_single.training_log.getLog()
        
        # Create feedback query for LLM
        feedback_query = f"""
Based on this training result, recommend improved parameters to achieve better accuracy (target: {target_accuracy*100}%).

Current Results:
- Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)
- Training Log: {json.dumps(training_log, indent=2)}
- Current Parameters: {json.dumps(trainer_single.getParams(), indent=2)}

Analysis Request:
1. Analyze what went wrong or could be improved
2. Suggest specific parameter changes to improve accuracy
3. Consider: learning rates, batch size, epochs, model architecture, training strategy
4. IMPORTANT: Consider image size optimization:
   - Use None for custom_img_size to enable auto-bucketing (recommended for most cases)
   - Larger images improve accuracy but increase training time and memory usage

Please provide your response in the following format:

EXPLANATION:
[Explain what you observed from the training results, what issues you identified, and why you're making specific changes. Be detailed about your reasoning.]

PARAMETERS:
[Return ONLY a valid Python dictionary with ONLY these allowed parameters:]
- base_model_name (EfficientNetB0, EfficientNetB1, etc.) THIS SHOULD ONLY BE CHANGED AS A LAST RESORT
- batch_size (integer)
- initial_learning_rate (float)
- fine_tune_learning_rate (float)
- initial_epochs (integer)
- fine_tune_epochs (integer)
- dual_stage (boolean)
- custom_img_size (list: [height, width] or None for auto-bucketing)
- unfreeze_percent (float 0.0-1.0)

Do NOT include img_size, dataset_path, or any other parameters.

IMPORTANT: Use Python syntax (True/False not true/false, None not null)
Analyze the training results and recommend parameter changes that will improve accuracy.

Focus on changes that will most likely improve accuracy based on the training history.
"""

        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            try:
                print("🤖 Querying LLM for parameter improvements...")
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": feedback_query}],
                    temperature=0.1
                )
                
                llm_response = response.choices[0].message.content
                print("✅ LLM feedback received")
                
                # Extract explanation first
                explanation_match = re.search(r'EXPLANATION:\s*(.*?)\s*PARAMETERS:', llm_response, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                    print(f"\n💡 LLM Analysis & Reasoning:")
                    print("="*60)
                    print(explanation)
                    print("="*60)
                else:
                    print("\n⚠️ No explanation found in LLM response")
                
                # Extract improved parameters
                dict_match = re.search(r'\{[^}]*\}', llm_response, re.DOTALL)
                if dict_match:
                    dict_str = dict_match.group()
                    dict_str = dict_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                    improved_params = eval(dict_str)
                    
                    print(f"\n🔧 LLM Suggested Parameter Changes:")
                    print(json.dumps(improved_params, indent=2))
                    
                    # Filter out invalid parameters that ImgClassTrainer doesn't accept
                    valid_params = {}
                    valid_keys = ['base_model_name', 'batch_size', 'initial_learning_rate', 
                                'fine_tune_learning_rate', 'initial_epochs', 'fine_tune_epochs', 
                                'dual_stage', 'custom_img_size', 'unfreeze_percent']
                    
                    for key, value in improved_params.items():
                        if key in valid_keys:
                            # Convert list to tuple for custom_img_size if needed
                            if key == 'custom_img_size' and isinstance(value, list):
                                valid_params[key] = tuple(value)
                            else:
                                valid_params[key] = value
                        else:
                            print(f"⚠️ Skipping invalid parameter: {key}")
                    
                    print(f"\n🔧 Filtered Valid Parameters:")
                    print(json.dumps(valid_params, indent=2))
                    
                    # Show parameter changes
                    current_params = trainer_single.getParams()
                    print(f"\n📊 Parameter Changes Made:")
                    print("-" * 50)
                    changes_made = False
                    for key in valid_params:
                        old_val = current_params.get(key, "N/A")
                        new_val = valid_params[key]
                        if old_val != new_val:
                            print(f"  {key:20}: {old_val} → {new_val}")
                            changes_made = True
                    
                    if not changes_made:
                        print("  No parameter changes detected")
                    print("-" * 50)
                    
                    # Create new trainer with improved parameters
                    trainer_single = ImgClassTrainer(
                        dataset_path=dataset_path,
                        **valid_params
                    )
                    print(f"✅ New trainer created with improved parameters")
                    
                else:
                    print("❌ Could not extract improved parameters from LLM response")
                    print("🔄 Continuing with current parameters...")
                    
            except Exception as e:
                print(f"❌ Error getting LLM feedback: {e}")
                print("🔄 Continuing with current parameters...")
        else:
            print("❌ No OpenAI API key - cannot get LLM feedback")
            print("🔄 Continuing with current parameters...")
    
    iteration += 1
    
    # Small delay between iterations
    if iteration <= max_iterations:
        time.sleep(2)

# Final summary
print(f"\n{'='*60}")
print(f"🏁 TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Final accuracy: {trainer_single.getMetrics().get('accuracy', 0.0):.4f}")
print(f"Total iterations: {iteration-1}")
print(f"Target achieved: {'✅ YES' if trainer_single.getMetrics().get('accuracy', 0.0) >= target_accuracy else '❌ NO'}")

# Show final training log
print(f"\n📋 Final Training Log:")
trainer_single.training_log.show()

# Save final log
trainer_single.save_training_log()