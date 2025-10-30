import logging
import sys

from datetime import datetime
from pathlib import Path
from typing import Optional


class Logger:
    """
    Standard Python logging utility for orchestration and status messages.

    This logger writes to both console and file, providing structured logging
    for the benchmarking pipeline components.
    """

    logger: Optional["Logger"] = None

    def __init__(
        self,
        logs_path: str,
        name: str = "TempusBench",
        enable_logging: bool = True,
        console_logging: bool = True,
        file_logging: bool = True,
        console_log_level: str = "INFO",
        file_log_level: str = "DEBUG",
    ):
        """
        Initialize logger with configuration.

        Args:
            logs_path: Directory to write log files
            name: Name for the logger instance
            enable_logging: Master switch for all logging (if False, no logging occurs)
            console_logging: Whether to log to console
            file_logging: Whether to log to file
            console_log_level: Console logging level (DEBUG, INFO, WARNING, ERROR)
            file_log_level: File logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.enable_logging = enable_logging
        self.console_logging = console_logging
        self.file_logging = file_logging
        self.console_log_level = console_log_level
        self.file_log_level = file_log_level

        # Create log directory if file logging is enabled
        if enable_logging and file_logging:
            Path(logs_path).mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Console handler (configurable level) - only if enabled
        if enable_logging and console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            # Convert string level to logging constant
            log_level = getattr(logging, console_log_level.upper(), logging.INFO)
            console_handler.setLevel(log_level)
            console_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # File handler (configurable level) - only if enabled
        if enable_logging and file_logging:
            log_file = Path(logs_path) / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            # Convert string level to logging constant
            log_level = getattr(logging, file_log_level.upper(), logging.DEBUG)
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

        # Set the class instance
        Logger.logger = self

    def _should_log(self) -> bool:
        """Check if logging should occur."""
        return self.enable_logging

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
