#!/usr/bin/env python3
"""
Logging Utility for Subprocess Debug Output
Provides centralized logging for all subprocess operations.
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, session_id: str = None, level: int = logging.DEBUG):
    """
    Setup logger for subprocess with file and console output.

    Args:
        name: Logger name (e.g., 'upload_handler', 'dataset_processor')
        session_id: Optional session ID for session-specific logs
        level: Logging level (default: DEBUG)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler - session-specific if session_id provided
    if session_id:
        log_filename = LOGS_DIR / f"{name}_{session_id}.log"
    else:
        log_filename = LOGS_DIR / f"{name}.log"

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for real-time output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Less verbose for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

def setup_llm_logger(session_id: str = None, level: int = logging.DEBUG):
    """
    Setup specialized logger for LLM input/output debugging.

    Args:
        session_id: Optional session ID for session-specific logs
        level: Logging level (default: DEBUG)
    """
    logger_name = "llm_debug"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create detailed formatter for LLM logs
    llm_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler - session-specific if session_id provided
    if session_id:
        log_filename = LOGS_DIR / f"llm_debug_{session_id}.log"
    else:
        log_filename = LOGS_DIR / f"llm_debug.log"

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level)
    file_handler.setFormatter(llm_formatter)
    logger.addHandler(file_handler)

    # Console handler for real-time output (less verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings/errors in console
    console_handler.setFormatter(llm_formatter)
    logger.addHandler(console_handler)

    return logger

def get_session_logs(session_id: str, limit: int = 100, include_llm_debug: bool = False):
    """
    Get recent logs for a specific session.

    Args:
        session_id: Session ID to get logs for
        limit: Maximum number of log lines to return
        include_llm_debug: Whether to include LLM debug logs (default: False for frontend)
    """
    logs = []
    log_files = [
        LOGS_DIR / f"upload_handler_{session_id}.log",
        LOGS_DIR / f"dataset_processor_{session_id}.log"
    ]
    
    # Only include LLM debug logs if explicitly requested (not for frontend)
    if include_llm_debug:
        log_files.append(LOGS_DIR / f"llm_debug_{session_id}.log")

    for log_file in log_files:
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    logs.extend(lines[-limit:])
            except Exception as e:
                logs.append(f"Error reading {log_file}: {e}")

    return logs[-limit:]  # Return the most recent logs

def get_llm_debug_logs(session_id: str, limit: int = 50):
    """
    Get LLM debug logs for a specific session (for debugging purposes only).

    Args:
        session_id: Session ID to get logs for
        limit: Maximum number of log lines to return
    """
    logs = []
    log_file = LOGS_DIR / f"llm_debug_{session_id}.log"
    
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                logs.extend(lines[-limit:])
        except Exception as e:
            logs.append(f"Error reading LLM debug log: {e}")
    else:
        logs.append("No LLM debug logs found for this session")

    return logs

def cleanup_old_logs(days: int = 7):
    """
    Clean up log files older than specified days.

    Args:
        days: Number of days to keep logs (default: 7)
    """
    import time

    cutoff_time = time.time() - (days * 24 * 60 * 60)

    for log_file in LOGS_DIR.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            try:
                log_file.unlink()
                print(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                print(f"Error cleaning up {log_file}: {e}")

if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test_logger", "test_session")
    logger.info("Test log message")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
