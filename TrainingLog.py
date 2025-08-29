import json
from datetime import datetime
import os

class TrainingLog:
    def __init__(self):
        self.log = {}
        self.current_iteration = 1

    def addEntry(self, params, logs, test):
        """Legacy method for backward compatibility - assumes single stage training."""
        # create a new entry dict for the current iteration
        self.log[f"iteration_{self.current_iteration}"] = {
            "params": params,
            "logs": logs,
            "test": test,
            "timestamp": datetime.now().isoformat(),
            "training_type": "single_stage"
        }
        self.current_iteration += 1

    def addTwoStageEntry(self, params, stage1_logs, stage2_logs, test_metrics):
        """Add entry for two-stage training with separate logs for each stage."""
        self.log[f"iteration_{self.current_iteration}"] = {
            "params": params,
            "stage1_logs": stage1_logs,
            "stage2_logs": stage2_logs,
            "test_metrics": test_metrics,
            "timestamp": datetime.now().isoformat(),
            "training_type": "two_stage"
        }
        self.current_iteration += 1


    def getLog(self):
        return self.log

    def json(self, pretty=True):
        if pretty:
            return json.dumps(self.log, indent=4)
        return json.dumps(self.log)
    def show(self):
        print(self.json(pretty=True))
    def save(self, filepath=f'training_log{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'):
        with open(filepath, 'w') as f:
            f.write(self.json())