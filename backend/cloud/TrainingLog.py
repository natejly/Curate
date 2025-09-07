import json
from datetime import datetime
import os
import sys

# Add backend root to sys.path for flexible imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class TrainingLog:
    def __init__(self):
        self.log = {}
        self.current_iteration = 1

    def addEntry(self, params, logs, test, ai_recommendations=None, optimization_iteration=None):
        """Legacy method for backward compatibility - assumes single stage training."""
        # create a new entry dict for the current iteration
        entry = {
            "params": params,
            "logs": logs,
            "test": test,
            "timestamp": datetime.now().isoformat(),
            "training_type": "single_stage"
        }
        
        # Add AI recommendations if provided
        if ai_recommendations:
            entry["ai_advisor"] = ai_recommendations
            
        # Add optimization iteration info if provided
        if optimization_iteration is not None:
            entry["optimization_iteration"] = optimization_iteration
            entry["is_optimization"] = True
        else:
            entry["is_optimization"] = False
            
        self.log[f"iteration_{self.current_iteration}"] = entry
        self.current_iteration += 1

    def addTwoStageEntry(self, params, stage1_logs, stage2_logs, test_metrics, ai_recommendations=None, optimization_iteration=None):
        """Add entry for two-stage training with separate logs for each stage."""
        entry = {
            "params": params,
            "stage1_logs": stage1_logs,
            "stage2_logs": stage2_logs,
            "test_metrics": test_metrics,
            "timestamp": datetime.now().isoformat(),
            "training_type": "two_stage"
        }
        
        # Add AI recommendations if provided
        if ai_recommendations:
            entry["ai_advisor"] = ai_recommendations
            
        # Add optimization iteration info if provided
        if optimization_iteration is not None:
            entry["optimization_iteration"] = optimization_iteration
            entry["is_optimization"] = True
        else:
            entry["is_optimization"] = False
            
        self.log[f"iteration_{self.current_iteration}"] = entry
        self.current_iteration += 1

    def get_log_data(self):
        """Get training log data in a format suitable for AI optimization analysis."""
        # Convert any non-JSON serializable types to JSON-compatible types
        def convert_value(obj):
            if hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'numpy'):  # tensorflow types
                return obj.numpy().item()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        return convert_value(self.log)

    def json(self, pretty=True):
        # Convert any non-JSON serializable types to JSON-compatible types
        def convert_value(obj):
            if hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif hasattr(obj, 'numpy'):  # tensorflow types
                return obj.numpy().item()
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        serializable_log = convert_value(self.log)
        
        if pretty:
            return json.dumps(serializable_log, indent=4)
        return json.dumps(serializable_log)
    def show(self):
        print(self.json(pretty=True))
    def save(self, filepath=f'training_log.json'):
        with open(filepath, 'w') as f:
            f.write(self.json())