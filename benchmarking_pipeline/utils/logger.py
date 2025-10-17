import logging
import sys

from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    """
    Standard Python logging utility for orchestration and status messages.

    This logger writes to both console and file, providing structured logging
    for the benchmarking pipeline components.
    """

    def __init__(self, logs_dir: str, name: str = "Logger"):
        """
        Initialize logger with configuration.

        Args:
            logs_dir: Directory to write log files
            name: Name for the logger instance
        """
        self.name = name

        # Create log directory
        Path(logs_dir).mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Console handler (shows INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)

        # File handler (shows DEBUG and above)
        log_file = Path(logs_dir) / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_format)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def info(self, message: str):
        """Log an informational message."""
        self.logger.info(f"[INFO] {message}")

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(f"[WARNING] {message}")

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(f"[ERROR] {message}")

    def success(self, message: str):
        """Log a success message."""
        self.logger.info(f"[SUCCESS] {message}")

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(f"[DEBUG] {message}")

    def progress(self, message: str):
        """Log a progress message."""
        self.logger.info(f"[PROGRESS] {message}")
