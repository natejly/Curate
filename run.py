import os
import openai
import time
import json
import re
import argparse
from datetime import datetime
from dotenv import load_dotenv
from ImgClassTrain import ImgClassTrainer
from ImgClassData import ImgClassData
from Prompts.training_prompts import get_initial_parameter_prompt, get_feedback_prompt

# Load environment variables from .env file
load_dotenv()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-powered image classification training')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output')
    parser.add_argument('--task', type=str, default='train digits classifier', 
                       help='Description of the training task')
    parser.add_argument('--dataset', type=str, default='/Users/natejly/Desktop/Rice_Image_dataset',
                       help='Path to the dataset directory')
    parser.add_argument('--target-accuracy', type=float, default=0.99,
                       help='Target accuracy (0.0-1.0)')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Maximum number of training iterations')
    return parser.parse_args()


def debug_print(message, debug_enabled=False):
    """Print debug messages only if debug is enabled."""
    if debug_enabled:
        print(f"🐛 DEBUG: {message}")


def analyze_dataset(dataset_path, prompt, debug=False):
    """Analyze the dataset and return metadata."""
    debug_print("Starting dataset analysis", debug)
    print("📊 Analyzing dataset...")
    
    data_parser = ImgClassData(dataset_path, debug=debug)
    
    dataset_info = {
        "prompt": prompt,
        "img_size": data_parser.IMSIZE,
        "file_tree": json.loads(data_parser.json_tree),
        "dataset_splits": json.loads(data_parser.DS_split_tree),
        "classes": data_parser.classes,
        "num_classes": len(data_parser.classes)
    }
    
    debug_print(f"Dataset info: {dataset_info}", debug)
    
    print("✅ Dataset Analysis Complete:")
    print(f"   Task: {prompt}")
    print(f"   Image Size: {data_parser.IMSIZE}")
    print(f"   Classes: {data_parser.classes}")
    print(f"   Number of Classes: {len(data_parser.classes)}")
    
    return dataset_info, data_parser


def get_initial_parameters(dataset_info, dataset_path, debug=False):
    """Get initial parameters from LLM or use defaults."""
    debug_print("Getting initial parameters from LLM", debug)
    
    # Create query for GPT-4o using imported prompt function
    llm_query = get_initial_parameter_prompt(dataset_info)
    
    # Get OpenAI API key from environment
    api_key = os.getenv('OPENAI_API_KEY')

    if api_key:
        try:
            print("\n🤖 Querying GPT-4o for initial optimal parameters...")
            # Query GPT-4o for training parameters
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": llm_query}]
            )
            
            llm_response = response.choices[0].message.content
            print("✅ Initial LLM Response received")
            
            debug_print(f"LLM response: {llm_response}", debug)
            
            # Extract explanation first
            explanation_match = re.search(r'EXPLANATION:\s*(.*?)\s*PARAMETERS:', llm_response, re.DOTALL)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                print(f"\n💡 LLM Initial Parameter Reasoning:")
                print("="*60)
                print(explanation)
                print("="*60)
            else:
                debug_print("No explanation found in initial LLM response", debug)
            
            # Extract dictionary from LLM response
            dict_match = re.search(r'\{[^}]*\}', llm_response, re.DOTALL)
            if dict_match:
                dict_str = dict_match.group()
                dict_str = dict_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                recommended_params = eval(dict_str)
                
                debug_print(f"Parsed parameters: {recommended_params}", debug)
                
                print("\n📋 AI-Recommended Initial Parameters:")
                print(json.dumps(recommended_params, indent=2))
                
                # Create trainer with LLM-recommended parameters
                trainer = ImgClassTrainer(
                    dataset_path=dataset_path,
                    **recommended_params
                )
                print(f"\n✅ Initial trainer created with AI recommendations!")
                print(f"   Training mode: {trainer.get_training_mode()}")
                return trainer
            else:
                raise ValueError("Could not extract parameters from LLM response")
                
        except Exception as e:
            print(f"❌ Error with initial LLM query: {e}")
            debug_print(f"LLM error details: {str(e)}", debug)
            
    # Create trainer with default parameters if LLM fails
    print("\n⚙️ Creating trainer with default parameters...")
    trainer = ImgClassTrainer(
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
    return trainer


def extract_accuracy(metrics, debug=False):
    """Extract accuracy from training metrics."""
    debug_print(f"Raw metrics: {metrics}", debug)
    
    final_accuracy = 0.0
    if metrics:
        for key in ['accuracy', 'compile_metrics', 'val_accuracy']:
            if key in metrics:
                final_accuracy = metrics[key]
                debug_print(f"Using accuracy from key: {key}", debug)
                break
    
    return final_accuracy


def run_training_iteration(trainer, iteration, max_iterations, target_accuracy, 
                          prompt, dataset_info, debug=False):
    """Run a single training iteration and return results."""
    print(f"\n{'='*60}")
    print(f"🚀 ITERATION {iteration}/{max_iterations}")
    print(f"{'='*60}")
    
    # Show current parameters
    current_params = trainer.getParams()
    print(f"📊 Current Parameters:")
    print(json.dumps(current_params, indent=2))
    
    # Run training
    print(f"\n🏋️ Training with current parameters...")
    trainer.run()
    
    # Get final test accuracy
    metrics = trainer.getMetrics()
    final_accuracy = extract_accuracy(metrics, debug)
    
    print(f"\n📈 Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Check if target accuracy reached
    if final_accuracy >= target_accuracy:
        print(f"🎉 SUCCESS! Target accuracy of {target_accuracy*100}% reached!")
        print(f"✅ Training completed in {iteration} iteration(s)")
        return True, final_accuracy
    
    # Get LLM feedback if not at target and not last iteration
    if iteration < max_iterations:
        print(f"📊 Accuracy below target ({target_accuracy*100}%). Getting LLM feedback...")
        
        # Prepare training log for LLM analysis
        training_log = trainer.training_log.getLog()
        
        # Create feedback query for LLM using imported prompt function  
        feedback_query = get_feedback_prompt(
            final_accuracy=final_accuracy,
            target_accuracy=target_accuracy, 
            training_log=json.dumps(training_log, indent=2),
            current_params=json.dumps(current_params, indent=2),
            user_task=prompt,
            dataset_info=dataset_info
        )
        
        # Get improved parameters from LLM
        success = get_llm_feedback_and_update(trainer, feedback_query, debug)
        if not success:
            debug_print("Failed to get LLM feedback, continuing with current parameters", debug)
    
    return False, final_accuracy


def get_llm_feedback_and_update(trainer, feedback_query, debug=False):
    """Get LLM feedback and update trainer parameters."""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No OpenAI API key - cannot get LLM feedback")
        return False
        
    try:
        debug_print("Querying LLM for parameter improvements", debug)
        print("🤖 Querying LLM for parameter improvements...")
        
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": feedback_query}],
        )
        
        llm_response = response.choices[0].message.content
        print("✅ LLM feedback received")
        
        debug_print(f"LLM feedback response: {llm_response}", debug)
        
        # Extract explanation first
        explanation_match = re.search(r'EXPLANATION:\s*(.*?)\s*PARAMETERS:', llm_response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
            print(f"\n💡 LLM Analysis & Reasoning:")
            print("="*60)
            print(explanation)
            print("="*60)
        else:
            debug_print("No explanation found in LLM response", debug)
        
        # Extract improved parameters
        dict_match = re.search(r'\{[^}]*\}', llm_response, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group()
            dict_str = dict_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            improved_params = eval(dict_str)
            
            print(f"\n🔧 LLM Suggested Parameter Changes:")
            print(json.dumps(improved_params, indent=2))
            
            # Filter out invalid parameters
            valid_params = filter_valid_parameters(improved_params, debug)
            
            # Show parameter changes
            show_parameter_changes(trainer.getParams(), valid_params, debug)
            
            # Update existing trainer parameters (preserves training history)
            trainer.edit_config(**valid_params)
            print(f"✅ Trainer parameters updated with improved values")
            return True
            
        else:
            print("❌ Could not extract improved parameters from LLM response")
            return False
            
    except Exception as e:
        print(f"❌ Error getting LLM feedback: {e}")
        debug_print(f"LLM feedback error details: {str(e)}", debug)
        return False


def filter_valid_parameters(params, debug=False):
    """Filter out invalid parameters that ImgClassTrainer doesn't accept."""
    valid_params = {}
    valid_keys = ['base_model_name', 'batch_size', 'initial_learning_rate', 
                  'fine_tune_learning_rate', 'initial_epochs', 'fine_tune_epochs', 
                  'dual_stage', 'custom_img_size', 'unfreeze_percent']
    
    for key, value in params.items():
        if key in valid_keys:
            # Convert list to tuple for custom_img_size if needed
            if key == 'custom_img_size' and isinstance(value, list):
                valid_params[key] = tuple(value)
            else:
                valid_params[key] = value
        else:
            print(f"⚠️ Skipping invalid parameter: {key}")
            debug_print(f"Filtered out parameter: {key} = {value}", debug)
    
    debug_print(f"Valid parameters: {valid_params}", debug)
    print(f"\n🔧 Filtered Valid Parameters:")
    print(json.dumps(valid_params, indent=2))
    
    return valid_params


def show_parameter_changes(current_params, new_params, debug=False):
    """Display what parameters changed between iterations."""
    print(f"\n📊 Parameter Changes Made:")
    print("-" * 50)
    changes_made = False
    
    for key in new_params:
        old_val = current_params.get(key, "N/A")
        new_val = new_params[key]
        if old_val != new_val:
            print(f"  {key:20}: {old_val} → {new_val}")
            changes_made = True
            debug_print(f"Parameter change: {key} changed from {old_val} to {new_val}", debug)
    
    if not changes_made:
        print("  No parameter changes detected")
        debug_print("No parameter changes detected", debug)
    print("-" * 50)


def main():
    """Main training loop with AI-guided optimization."""
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"🤖 AI-Powered Image Classification Training")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Target accuracy: {args.target_accuracy*100}%")
    print(f"Max iterations: {args.max_iterations}")
    
    # Analyze dataset
    dataset_info, data_parser = analyze_dataset(args.dataset, args.task, args.debug)
    
    # Get initial trainer with parameters
    trainer = get_initial_parameters(dataset_info, args.dataset, args.debug)
    
    print(f"\n🎯 Initial Configuration Ready:")
    print(json.dumps(trainer.getParams(), indent=2))
    
    # Training loop
    print("🔄 Starting iterative LLM-guided training...")
    print(f"Target accuracy: {args.target_accuracy*100}%")
    print(f"Max iterations: {args.max_iterations}")
    
    for iteration in range(1, args.max_iterations + 1):
        success, final_accuracy = run_training_iteration(
            trainer, iteration, args.max_iterations, args.target_accuracy,
            args.task, dataset_info, args.debug
        )
        
        if success:
            break
    else:
        # If we exit the loop without breaking (max iterations reached)
        print(f"\n⏰ Maximum iterations ({args.max_iterations}) reached.")
        print(f"📈 Final accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        if final_accuracy < args.target_accuracy:
            print(f"🎯 Target accuracy of {args.target_accuracy*100}% not reached.")
        
    print("\n✅ Training process completed!")
    
    # Final results
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    final_metrics = trainer.getMetrics()
    final_acc = extract_accuracy(final_metrics, args.debug)
    print(f"Final accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"Target accuracy: {args.target_accuracy:.4f} ({args.target_accuracy*100:.2f}%)")
    print(f"Target achieved: {'✅ YES' if final_acc >= args.target_accuracy else '❌ NO'}")
    
    # Show training log
    trainer.training_log.show()
    trainer.training_log.sa


if __name__ == "__main__":
    main()
