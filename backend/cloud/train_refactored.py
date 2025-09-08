#!/usr/bin/env python3
"""
Refactored SageMaker Training Script for Image Classification

This refactored version provides:
- Clean separation of concerns
- Modular architecture
- Better error handling
- Improved maintainability
"""

import os
import sys

# Add directories to Python path for SageMaker environment
sys.path.insert(0, os.path.dirname(__file__))  # Add current directory (cloud)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # Add backend directory  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # Add repository root

print(f"Python path: {sys.path[:5]}")  # Debug: Show first 5 path entries
print(f"Current working directory: {os.getcwd()}")  # Debug: Show current directory

from config import ConfigManager
from training_orchestrator import run_training_pipeline


def main():
    """
    Main entry point for training pipeline.
    
    This simplified main function delegates all work to the TrainingOrchestrator,
    providing clean separation of concerns and better error handling.
    """
    try:
        # Parse configuration from command line
        config = ConfigManager.parse_args()
        
        # Run the complete training pipeline
        run_training_pipeline(config)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
