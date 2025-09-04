#!/usr/bin/env python3
"""
Example script showing how to fetch and export trained models from S3.

This script demonstrates the workflow:
1. Training saves both .keras and .onnx models to S3 organized by session
2. User can later fetch specific models for export/inference using session ID
"""

import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from trainio import fetch_model_from_s3, fetch_session_model, list_session_models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example of fetching models from S3 using session-based approach."""

    # Example session ID (this would be from your training run)
    session_id = "training_session_001"

    print("🔍 Model Export Examples")
    print("=" * 50)

    # Method 1: Fetch by session ID (recommended)
    print("\n📋 Method 1: Fetch by Session ID")
    try:
        # Fetch the latest Keras model for this session
        local_path, model_info = fetch_session_model(session_id, 'keras', download_dir="./exported_models")

        print("✅ Keras model fetched successfully!")
        print(f"📁 Local path: {local_path}")
        print(f"📦 File size: {model_info['size_mb']:.2f} MB")
        print(f"📅 Last modified: {model_info['last_modified']}")
        print(f"🎯 Model ready for TensorFlow inference!")

        # Fetch ONNX model as well
        onnx_path, onnx_info = fetch_session_model(session_id, 'onnx', download_dir="./exported_models")
        print("\n✅ ONNX model fetched successfully!")
        print(f"📁 Local path: {onnx_path}")
        print(f"📦 File size: {onnx_info['size_mb']:.2f} MB")

    except Exception as e:
        print(f"❌ Failed to fetch session models: {str(e)}")
        print("💡 Make sure the session ID matches your training run")

    # Method 2: List all available models for a session
    print("\n📋 Method 2: List All Models for Session")
    try:
        session_models = list_session_models(session_id=session_id)
        if session_id in session_models:
            print(f"📂 Models available for session '{session_id}':")
            for model in session_models[session_id]:
                print(f"  • {model['format'].upper()}: {model['filename']} ({model['size_mb']:.2f} MB)")
        else:
            print(f"❌ No models found for session '{session_id}'")
    except Exception as e:
        print(f"❌ Failed to list session models: {str(e)}")

    # Method 3: Fetch by direct S3 path (legacy method)
    print("\n📋 Method 3: Fetch by Direct S3 Path")
    try:
        # Example S3 paths (these would come from training logs)
        example_s3_paths = {
            'keras': f's3://curate-sagemaker-bucket-123456789012/curate/models/sessions/{session_id}/20241215_143052_keras_model.keras',
            'onnx': f's3://curate-sagemaker-bucket-123456789012/curate/models/sessions/{session_id}/20241215_143052_onnx_model.onnx'
        }

        s3_path = example_s3_paths['keras']
        local_path = fetch_model_from_s3(s3_path, download_dir="./exported_models")
        print(f"✅ Model fetched from direct path: {local_path}")

    except Exception as e:
        print(f"❌ Failed to fetch by direct path: {str(e)}")

    print("\n🎯 Export Complete!")
    print("Your models are now ready for:")
    print("  • Deployment to production")
    print("  • Further fine-tuning")
    print("  • Model evaluation")
    print("  • Integration with other systems")

if __name__ == "__main__":
    main()
