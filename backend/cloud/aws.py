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
        # Optional: use a Git repo as source. Disabled by default to use local source_dir
        self.git_config = None
        self.entrypoint = "cloud/train.py"
        self.role = "arn:aws:iam::974703727033:role/SageMakerExecutionRole"
        self.s3_path = None
        self.base_job_name = "curate-job"
    def set_base_job_name(self, name):
        self.base_job_name = name
    def start_sagemaker_executor(self, instance_type="ml.m5.large", instance_count=1, hyperparameters=None, output_path=None, return_estimator=False, wait=True):
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
        
        estimator_kwargs = dict(
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
            output_path=output_path,
            base_job_name="curate-tf-job",
            sagemaker_session=self.session
        )
        if self.git_config:
            estimator_kwargs["git_config"] = self.git_config
        estimator = TensorFlow(**estimator_kwargs)

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
