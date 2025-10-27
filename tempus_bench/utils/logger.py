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

    def __init__(self, logs_path: str, name: str = "TempusBench", console_logging: bool = True, file_logging: bool = True):
        """
        Initialize logger with configuration.

        Args:
            logs_path: Directory to write log files
            name: Name for the logger instance
            console_logging: Whether to log to console
            file_logging: Whether to log to file
        """
        self.name = name
        self.logs_path = logs_path
        self.console_logging = console_logging
        self.file_logging = file_logging

        # Create log directory if file logging is enabled
        if file_logging:
            Path(logs_path).mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Console handler (shows INFO and above) - only if enabled
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # File handler (shows DEBUG and above) - only if enabled
        if file_logging:
            log_file = Path(logs_path) / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _should_log(self) -> bool:
        """Check if logging should occur (either console or file)."""
        return self.console_logging or self.file_logging

    def info(self, module: str, message: str):
        """Log an informational message with module context."""
        if self._should_log():
            self.logger.info(f"[{module}] {message}")

    def warning(self, module: str, message: str):
        """Log a warning message with module context."""
        if self._should_log():
            self.logger.warning(f"[{module}] {message}")

    def error(self, module: str, message: str):
        """Log an error message with module context."""
        if self._should_log():
            self.logger.error(f"[{module}] {message}")

    def success(self, module: str, message: str):
        """Log a success message with module context."""
        if self._should_log():
            self.logger.info(f"[{module}] SUCCESS: {message}")

    def debug(self, module: str, message: str):
        """Log a debug message with module context."""
        if self._should_log():
            self.logger.debug(f"[{module}] {message}")

    def progress(self, module: str, message: str):
        """Log a progress message with module context."""
        if self._should_log():
            self.logger.info(f"[{module}] PROGRESS: {message}")

# Global logger instance
_global_logger = None

def get_logger(logs_path: str, console_logging: Optional[bool] = None, file_logging: Optional[bool] = None) -> Logger:
    """
    Get or create the global logger instance.

    Args:
        logs_path: Directory to write log files
        console_logging: Whether to log to console
        file_logging: Whether to log to file

    Returns:
        Logger: Global logger instance
    """
    global _global_logger
    if _global_logger is None or _global_logger.logs_path != logs_path:
        _global_logger = Logger(logs_path, console_logging=console_logging, file_logging=file_logging)
    return _global_logger
