"""
Logging setup for training pipeline.
"""

import logging
import boto3
from datetime import datetime
from typing import Optional
import os


class CloudWatchHandler(logging.Handler):
    """Custom CloudWatch logging handler for real-time streaming."""
    
    def __init__(self, log_group: str, log_stream: str, region: str = 'us-east-1'):
        super().__init__()
        self.log_group = log_group
        self.log_stream = log_stream
        self.logs_client = boto3.client('logs', region_name=region)
        self.sequence_token = None
        self.buffer = []
        self.max_buffer_size = 1  # Send logs immediately for real-time streaming
        
    def emit(self, record):
        """Emit a log record to CloudWatch."""
        try:
            log_message = self.format(record)
            timestamp = int(datetime.now().timestamp() * 1000)
            
            log_event = {
                'timestamp': timestamp,
                'message': log_message
            }
            
            self.buffer.append(log_event)
            
            # Send immediately for real-time streaming
            if len(self.buffer) >= self.max_buffer_size:
                self.flush_buffer()
                
        except Exception as e:
            print(f"CloudWatch logging error: {e}")
            
    def flush_buffer(self):
        """Flush buffered log events to CloudWatch."""
        if not self.buffer:
            return
            
        try:
            kwargs = {
                'logGroupName': self.log_group,
                'logStreamName': self.log_stream,
                'logEvents': self.buffer
            }
            
            if self.sequence_token:
                kwargs['sequenceToken'] = self.sequence_token
            
            response = self.logs_client.put_log_events(**kwargs)
            self.sequence_token = response.get('nextSequenceToken')
            self.buffer = []
            
        except Exception as e:
            print(f"CloudWatch buffer flush error: {e}")
            # Reset buffer to prevent memory buildup
            self.buffer = []
            
    def close(self):
        """Close handler and flush remaining logs."""
        self.flush_buffer()
        super().close()


class StreamToLogger:
    """Redirect stdout/stderr to logger."""
    
    def __init__(self, level: int):
        self.level = level
        
    def write(self, message: str):
        msg = message.rstrip()
        if msg:
            logging.getLogger().log(self.level, msg)
            
    def flush(self):
        pass


class LoggingManager:
    """Manages logging setup for training pipeline."""
    
    @staticmethod
    def setup_basic_logging():
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger()
    
    @staticmethod
    def setup_cloudwatch_logging(session_id: Optional[str] = None) -> Optional[CloudWatchHandler]:
        """Set up CloudWatch logging with region detection."""
        log_group = "/curate/training"
        log_stream = session_id or "default-stream"
        
        try:
            # Detect AWS region
            region = (
                os.environ.get('AWS_REGION') or 
                os.environ.get('AWS_DEFAULT_REGION') or 
                boto3.session.Session().region_name or 
                'us-east-1'
            )
            
            print(f"[DEBUG] Setting up CloudWatch logging with region: {region}")
            logs_client = boto3.client('logs', region_name=region)
            
            # Create log group if it doesn't exist
            try:
                logs_client.create_log_group(logGroupName=log_group)
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Create log stream if it doesn't exist
            try:
                logs_client.create_log_stream(
                    logGroupName=log_group, 
                    logStreamName=log_stream
                )
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Create and configure handler
            cw_handler = CloudWatchHandler(log_group, log_stream, region)
            cw_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            cw_handler.setFormatter(formatter)
            
            return cw_handler
            
        except Exception as e:
            print(f"Failed to setup CloudWatch logging: {e}")
            return None
    
    @staticmethod
    def redirect_stdout_stderr():
        """Redirect stdout and stderr to logger."""
        import sys
        sys.stdout = StreamToLogger(logging.INFO)
        sys.stderr = StreamToLogger(logging.ERROR)
    
    @staticmethod
    def setup_tensorflow_logging():
        """Configure TensorFlow logging to reduce verbosity."""
        import tensorflow as tf
        import warnings
        
        # Suppress TensorFlow verbose output
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings and info
        
        # Suppress TensorFlow memory allocation warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        
        # Configure TensorFlow memory growth to avoid allocation warnings
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return gpus
        except RuntimeError:
            # Memory growth must be set before GPUs have been initialized
            pass
        
        return []
    
    @classmethod
    def setup_complete_logging(cls, session_id: Optional[str] = None) -> logging.Logger:
        """Setup complete logging configuration for training."""
        # Setup basic logging
        logger = cls.setup_basic_logging()
        
        # Setup TensorFlow logging and get GPU info
        gpus = cls.setup_tensorflow_logging()
        logger.info(f"Using GPU: {gpus}" if gpus else "No GPU found, using CPU")
        
        # Setup CloudWatch logging
        cw_handler = cls.setup_cloudwatch_logging(session_id)
        if cw_handler:
            logger.addHandler(cw_handler)
            logger.info(f"Custom CloudWatch logging started for session {session_id}")
            print(f"[DEBUG] Custom CloudWatch logging is ACTIVE")
        else:
            logger.warning("CloudWatch logging setup failed, using default logging")
            print(f"[DEBUG] Custom CloudWatch logging FAILED, using SageMaker default logs")
        
        # Redirect stdout/stderr to logger
        cls.redirect_stdout_stderr()
        
        return logger
