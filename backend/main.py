from cloud.aws import AWSHelper
import os

# Load environment variables from .env file if it exists
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize AWS helper
aws_helper = AWSHelper("curate-sagemaker-bucket-123456789012")

# Upload dataset to S3
aws_helper.upload_zip("/Users/natejly/Desktop/Rice_Image_Dataset", "curate/datasets/")
aws_helper.set_base_job_name("riceimgs")
# Start SageMaker training with AI advisor enabled
hyperparameters = {
    'use_ai_advisor': '',  # Enable AI advisor
    'apply_recommendations': '',  # Auto-apply recommendations
    'save_recommendations': '',  # Save recommendations to file
    'epochs': 10,  # Override default epochs
    'batch_size': 32  # Override default batch size
}

# OpenAI API key will be passed as environment variable (more secure)
if OPENAI_API_KEY:
    print("✅ OpenAI API key found - AI advisor will be enabled")
else:
    print("⚠️ OpenAI API key not found - AI advisor will be disabled")

aws_helper.start_sagemaker_executor(
    instance_type="ml.g4dn.xlarge",  # CPU instance with higher availability
    instance_count=1, 
    hyperparameters=hyperparameters,
    output_path=f"s3://{aws_helper.bucket}/curate/output/",
    wait=False
)
