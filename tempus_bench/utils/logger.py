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

    def __init__(self, logs_dir: str, name: str = "BenchmarkPipeline"):
        """
        Initialize logger with configuration.

        Args:
            logs_dir: Directory to write log files
            name: Name for the logger instance
        """
        self.name = name
        self.logs_dir = logs_dir

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

    def info(self, module: str, message: str):
        """Log an informational message with module context."""
        self.logger.info(f"[{module}] {message}")

    def warning(self, module: str, message: str):
        """Log a warning message with module context."""
        self.logger.warning(f"[{module}] {message}")

    def error(self, module: str, message: str):
        """Log an error message with module context."""
        self.logger.error(f"[{module}] {message}")

    def success(self, module: str, message: str):
        """Log a success message with module context."""
        self.logger.info(f"[{module}] SUCCESS: {message}")

    def debug(self, module: str, message: str):
        """Log a debug message with module context."""
        self.logger.debug(f"[{module}] {message}")

    def progress(self, module: str, message: str):
        """Log a progress message with module context."""
        self.logger.info(f"[{module}] PROGRESS: {message}")

# Global logger instance
_global_logger = None

def get_logger(logs_dir: str) -> Logger:
    """
    Get or create the global logger instance.

    Args:
        logs_dir: Directory to write log files

    Returns:
        Logger: Global logger instance
    """
    global _global_logger
    if _global_logger is None or _global_logger.logs_dir != logs_dir:
        _global_logger = Logger(logs_dir)
    return _global_logger
