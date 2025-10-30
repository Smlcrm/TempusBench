"""
Unified logging utility for both standard Python logging and TensorBoard logging.

This logger combines standard Python logging (console/file) with TensorBoard logging
capabilities, providing a single interface for all logging needs in the benchmarking pipeline.
"""

import io
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorboard.plugins.hparams import api as hp
from typing import Optional

# Configure TensorFlow threading before import
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class LoggerManager:
    """
    Unified logging utility for orchestration, status messages, and TensorBoard logging.

    This logger provides:
    - Standard Python logging to console and/or file
    - TensorBoard logging for metrics, plots, and hyperparameters
    - Independent control over each logging type
    """

    logger: LoggerManager

    def __init__(
        self,
        logs_path: str,
        name: str = "TempusBench",
        enable_logging: bool = True,
        console_logging: bool = True,
        file_logging: bool = True,
        console_log_level: str = "INFO",
        file_log_level: str = "DEBUG",
        tf_logs_path: Optional[str] = None,
        tensorboard_logging: Optional[bool] = None,
    ):
        """
        Initialize logger with configuration for both standard and TensorBoard logging.

        Args:
            logs_path: Directory to write standard log files
            name: Name for the logger instance
            enable_logging: Master switch for standard logging (if False, no standard logging occurs)
            console_logging: Whether to log to console
            file_logging: Whether to log to file
            console_log_level: Console logging level (DEBUG, INFO, WARNING, ERROR)
            file_log_level: File logging level (DEBUG, INFO, WARNING, ERROR)
            tf_logs_path: Directory to write TensorBoard log files (optional, defaults to logs_path/tensorboard)
            tensorboard_logging: Whether to enable TensorBoard logging (optional)
        """
        self.name = name
        self.enable_logging = enable_logging
        self.console_logging = console_logging
        self.file_logging = file_logging
        self.console_log_level = console_log_level
        self.file_log_level = file_log_level
        self.tensorboard_logging = tensorboard_logging

        # Create log directory if file logging is enabled
        if enable_logging and file_logging:
            Path(logs_path).mkdir(parents=True, exist_ok=True)

        # Setup standard logger
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

        # Setup TensorBoard logging
        if tf_logs_path is None:
            # Default to tensorboard subdirectory of logs_path
            self.tf_logs_path = str(Path(logs_path).parent / "tensorboard")
        else:
            self.tf_logs_path = tf_logs_path

        if tensorboard_logging:
            os.makedirs(self.tf_logs_path, exist_ok=True)
            self.tf_writer = tf.summary.create_file_writer(self.tf_logs_path)
        else:
            self.tf_writer = None

    def _should_log(self) -> bool:
        """Check if standard logging should occur."""
        return self.enable_logging

    def _should_log_tensorboard(self) -> bool:
        """Check if TensorBoard logging should occur."""
        return self.tensorboard_logging and self.tf_writer is not None

    # ==================== Standard Logging Methods ====================

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

    # ==================== TensorBoard Logging Methods ====================

    def log_metrics(self, metrics, step, model_name=""):
        """
        Log evaluation metrics to TensorBoard.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): The current step (e.g., epoch, batch, or experiment ID).
            model_name (str, optional): A prefix for metric names to group them in TensorBoard.
        """
        if not self._should_log_tensorboard():
            return

        group_prefix = f"{model_name}/" if model_name else ""
        with self.tf_writer.as_default():
            for metric_name, value in metrics.items():
                if value is None or (
                    isinstance(value, (float, int)) and np.isnan(value)
                ):
                    continue

                if isinstance(value, dict):  # For nested results like quantile losses
                    for sub_metric, sub_value in value.items():
                        tag = f"{group_prefix}{metric_name}/{sub_metric}"
                        tf.summary.scalar(tag, sub_value, step=step)
                elif isinstance(value, (np.ndarray, list)):
                    # For arrays log each element and the mean
                    value_arr = np.asarray(value)
                    tf.summary.scalar(
                        f"{group_prefix}{metric_name}/mean",
                        np.nanmean(value_arr),
                        step=step,
                    )
                    for i, elem in enumerate(value_arr):
                        tf.summary.scalar(
                            f"{group_prefix}{metric_name}/series_{i}", elem, step=step
                        )
                elif isinstance(value, (float, np.floating, int)):
                    tf.summary.scalar(f"{group_prefix}{metric_name}", value, step=step)
                else:
                    # Skip unsupported metric types silently (TensorBoard-only logging)
                    pass
        self.tf_writer.flush()

    def log_figure(self, figure, tag: str, step: int):
        """
        Log a Matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure object
            tag (str): Tag for the figure
            step (int): Step number
        """
        if not self._should_log_tensorboard():
            return

        buf = io.BytesIO()
        figure.savefig(buf, format="png")
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        with self.tf_writer.as_default():
            tf.summary.image(tag, image, step=step)
        self.tf_writer.flush()

    def log_image_file(self, image_path: str, tag: str, step: int):
        """
        Log an image from disk to TensorBoard.

        Args:
            image_path (str): Path to the image file
            tag (str): Tag for the image
            step (int): Step number
        """
        if not self._should_log_tensorboard():
            return

        try:
            with open(image_path, "rb") as f:
                data = f.read()
            image = tf.image.decode_image(data, channels=4)
            image = tf.expand_dims(image, 0)
            with self.tf_writer.as_default():
                tf.summary.image(tag, image, step=step)
            self.tf_writer.flush()
        except Exception as e:
            # Silently skip failed image logging (TensorBoard-only logging)
            pass

    def log_training_progress(self, model_name, epoch, loss, val_loss=None, step=None):
        """
        Log training progress for real-time monitoring.

        Args:
            model_name (str): Name of the model being trained
            epoch (int): Current epoch number
            loss (float): Training loss
            val_loss (float, optional): Validation loss
            step (int, optional): Global step for TensorBoard
        """
        if not self._should_log_tensorboard():
            return

        if step is None:
            step = epoch

        # Log to TensorBoard
        with self.tf_writer.as_default():
            tf.summary.scalar(f"{model_name}/train_loss", loss, step=step)
            if val_loss is not None:
                tf.summary.scalar(f"{model_name}/val_loss", val_loss, step=step)
        self.tf_writer.flush()

    def log_hparams(self, hparams, metrics):
        """
        Log a set of hyperparameters and the resulting metrics for comparison
        in TensorBoard's HParams dashboard.

        Args:
            hparams (dict): Dictionary of hyperparameters used for the run
                            (e.g., {'learning_rate': 0.1, 'model': 'XGBoost'}).
            metrics (dict): Dictionary of final metrics for this run (e.g., {'mae': 12.3}).
        """
        if not self._should_log_tensorboard():
            return

        with self.tf_writer.as_default():
            # Sanitize hparams for TensorBoard (it has specific type requirements)
            sanitized_hparams = {
                k: v
                for k, v in hparams.items()
                if isinstance(v, (str, bool, int, float))
            }
            hp.hparams(sanitized_hparams)
            for metric_name, value in metrics.items():
                if isinstance(value, (float, np.floating, int)):
                    # Log final metrics in a way that HParams can read
                    tf.summary.scalar(f"hparams/{metric_name}", value, step=1)
        self.tf_writer.flush()

    def log_text(self, tag: str, text: str, step: int):
        """
        Log text to TensorBoard.

        Args:
            tag (str): Tag for the text
            text (str): Text content to log
            step (int): Step number
        """
        if not self._should_log_tensorboard():
            return

        with self.tf_writer.as_default():
            tf.summary.text(tag, text, step=step)
        self.tf_writer.flush()

    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value to TensorBoard.

        Args:
            tag (str): Tag for the scalar
            value (float): Scalar value to log
            step (int): Step number
        """
        if not self._should_log_tensorboard():
            return

        with self.tf_writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.tf_writer.flush()

    def close(self):
        """Closes the TensorBoard writer to ensure all data is written to disk."""
        if self.tf_writer is not None:
            self.tf_writer.close()
