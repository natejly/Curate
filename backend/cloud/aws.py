import boto3
import os
import time
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm  # install via: pip install tqdm
import shutil
import sagemaker
from sagemaker.tensorflow import TensorFlow

class AWSHelper:
    def __init__(self, bucket_name):
        self.dry_run = False
        self.bucket = bucket_name
        self.s3_client = boto3.client("s3")
        self.config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,   # 8 MB
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=20,  # allow up to 20 threads for multipart chunks
        use_threads=True
    )
        self.session = sagemaker.Session()
        self.git_config = {
            "repo": "https://github.com/natejly/curate.git",
            "branch": "main"
        }
        self.entrypoint = "cloud/train.py"
        self.role = "arn:aws:iam::974703727033:role/SageMakerExecutionRole"
        self.s3_path = None
        self.base_job_name = "curate-job"
    def set_base_job_name(self, name):
        self.base_job_name = name
    def start_sagemaker_executor(self, instance_type="ml.m5.large", instance_count=1, hyperparameters=None, output_path=None, return_estimator=False, wait=False):
        """Run training job on AWS SageMaker"""
        if hyperparameters is None:
            hyperparameters = {}
        if output_path is None:
            output_path = f"s3://{self.bucket}/curate/output/"

        # Pass S3 zip path to train.py via hyperparameters
        if self.s3_path:
            hyperparameters['zip_s3_path'] = self.s3_path

        # Pass environment variables to SageMaker
        environment = {}
        
        # Pass OpenAI API key as environment variable (secure)
        local_openai_key = os.getenv('OPENAI_API_KEY')
        if local_openai_key:
            environment['OPENAI_API_KEY'] = local_openai_key
            print("✅ OpenAI API key will be available in SageMaker environment")
        else:
            print("⚠️ No OpenAI API key found - AI advisor will be disabled")
        # Ensure AWS region is present for explicit boto3 clients
        local_region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
        if local_region:
            environment['AWS_REGION'] = local_region
            environment.setdefault('AWS_DEFAULT_REGION', local_region)
            print(f"✅ AWS region set for training container: {local_region}")
        
        estimator = TensorFlow(
            entry_point=self.entrypoint,
            source_dir="backend",  # Include all files in cloud directory
            # dependencies are handled by requirements.txt in source_dir
            role=self.role,
            instance_count=instance_count,
            instance_type=instance_type,
            framework_version="2.12",
            py_version="py310",
            hyperparameters=hyperparameters,
            environment=environment,  # Pass environment variables
            git_config=self.git_config,
            output_path=output_path,
            base_job_name="curate-tf-job",
            sagemaker_session=self.session
        )

        print(f"Starting SageMaker job with entrypoint '{self.entrypoint}' on {instance_count} x {instance_type}...")
        estimator.fit(wait=wait)
        print("SageMaker job started.")
        if return_estimator:
            return estimator

    def upload_zip(self, folder_path, s3_prefix=""):
        """
        Zip a local folder and upload the .zip file to S3.
        Much faster for many small files.

        :param folder_path: Local folder path (e.g., "data/")
        :param s3_prefix: Prefix inside the bucket (e.g., "curate/train/")
        """
        archive_name = os.path.basename(os.path.normpath(folder_path))
        zip_path = f"{archive_name}.zip"

        print(f"Zipping {folder_path} → {zip_path} ...")
        start_zip = time.time()
        shutil.make_archive(archive_name, "zip", folder_path)
        zip_size = os.path.getsize(zip_path)
        print(f"Zip complete in {time.time()-start_zip:.2f}s, size {zip_size/1e6:.2f} MB")

        # Step 2: Upload the .zip file
        s3_key = os.path.join(s3_prefix, os.path.basename(zip_path)).replace("\\", "/")

        print(f"Uploading archive to s3://{self.bucket}/{s3_key} ...")
        self.s3_path = f"s3://{self.bucket}/{s3_key}"
        start_upload = time.time()

        config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,   # 8 MB
            multipart_chunksize=8 * 1024 * 1024,
            max_concurrency=10,
            use_threads=True
        )

        with tqdm(total=zip_size, unit="B", unit_scale=True, desc="Uploading ZIP") as pbar:
            self.s3_client.upload_file(zip_path, self.bucket, s3_key, Config=config,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred))

        print(f"Upload complete in {time.time()-start_upload:.2f}s")

        # (Optional) clean up local zip
        os.remove(zip_path)
        print(f"Removed local zip: {zip_path}")

    def upload_file(self, file_path, s3_key):
        """
        Upload a single file to S3.

        :param file_path: Local file path
        :param s3_key: S3 key (path in bucket)
        """
        file_size = os.path.getsize(file_path)
        print(f"Uploading file {file_path} to s3://{self.bucket}/{s3_key} ({file_size/1e6:.2f} MB)...")

        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as pbar:
            self.s3_client.upload_file(
                file_path,
                self.bucket,
                s3_key,
                Config=self.config,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )

        print(f"File upload complete: s3://{self.bucket}/{s3_key}")

    def upload_directory(self, directory_path, s3_prefix):
        """
        Upload all files in a directory to S3, preserving the directory structure.

        :param directory_path: Local directory path
        :param s3_prefix: S3 prefix (folder path in bucket)
        """
        print(f"Uploading directory {directory_path} to s3://{self.bucket}/{s3_prefix}/")

        total_files = 0
        total_size = 0

        # First pass: count files and total size
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                total_files += 1

        print(f"Found {total_files} files ({total_size/1e6:.2f} MB total)")

        uploaded_files = 0
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Uploading") as pbar:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    # Create relative path from directory
                    relative_path = os.path.relpath(local_path, directory_path)
                    s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

                    self.s3_client.upload_file(
                        local_path,
                        self.bucket,
                        s3_key,
                        Config=self.config
                    )

                    file_size = os.path.getsize(local_path)
                    pbar.update(file_size)
                    uploaded_files += 1

        print(f"Directory upload complete: {uploaded_files} files uploaded to s3://{self.bucket}/{s3_prefix}/")
