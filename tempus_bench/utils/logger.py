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
        tensorboard_logging: bool = False,
    ):
        """
        Initialize logger with configuration for both standard and TensorBoard logging.

        Note: Logger and SummaryWriter objects are always created regardless of flag values.
        The actual logging behavior is controlled by `enable_logging` and `tensorboard_logging`.

        Args:
            logs_path: Directory to write standard log files
            name: Name for the logger instance
            enable_logging: Controls whether standard logging methods actually log (Logger is always created)
            console_logging: Whether to create console handler
            file_logging: Whether to create file handler
            console_log_level: Console logging level (DEBUG, INFO, WARNING, ERROR)
            file_log_level: File logging level (DEBUG, INFO, WARNING, ERROR)
            tf_logs_path: Directory to write TensorBoard log files (optional, defaults to logs_path/tensorboard)
            tensorboard_logging: Controls whether TensorBoard logging methods actually log (SummaryWriter is always created)
        """
        self.name = name
        self.enable_logging = enable_logging
        self.tensorboard_logging = tensorboard_logging

        # Always create log directory
        Path(logs_path).mkdir(parents=True, exist_ok=True)

        # Always setup standard logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Always create console handler (configurable level)
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            # Convert string level to logging constant
            log_level = getattr(logging, console_log_level.upper(), logging.INFO)
            console_handler.setLevel(log_level)
            console_format = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

        # Always create file handler (configurable level)
        if file_logging:
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

        # Always setup TensorBoard logging
        if tf_logs_path is None:
            # Default to tensorboard subdirectory of logs_path
            self.tf_logs_path = str(Path(logs_path).parent / "tensorboard")
        else:
            self.tf_logs_path = tf_logs_path

        # Always create TensorBoard writer
        os.makedirs(self.tf_logs_path, exist_ok=True)
        self.tf_writer = tf.summary.create_file_writer(self.tf_logs_path)

    def _should_log(self) -> bool:
        """
        Check if standard logging should occur.

        Returns:
            bool: True if standard logging is enabled, False otherwise.
        """
        return self.enable_logging

    def _should_log_tensorboard(self) -> bool:
        """
        Check if TensorBoard logging should occur.

        Returns:
            bool: True if TensorBoard logging is enabled, False otherwise.
        """
        return self.tensorboard_logging

    # ==================== Context Manager Methods ====================

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            LoggerManager: Returns self for use in with statements.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, ensuring all logs are flushed and resources closed.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.

        Returns:
            bool: False, indicating that exceptions should not be suppressed.
        """
        self.close()
        return False  # Don't suppress exceptions

    # ==================== Standard Logging Methods ====================

    def info(self, module: str, message: str):
        """
        Log an informational message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.info(f"[{module}] {message}")

    def warning(self, module: str, message: str):
        """
        Log a warning message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.warning(f"[{module}] {message}")

    def error(self, module: str, message: str):
        """
        Log an error message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.error(f"[{module}] {message}")

    def success(self, module: str, message: str):
        """
        Log a success message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.info(f"[{module}] SUCCESS: {message}")

    def debug(self, module: str, message: str):
        """
        Log a debug message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.debug(f"[{module}] {message}")

    def progress(self, module: str, message: str):
        """
        Log a progress message with module context.

        Args:
            module (str): Module name for context.
            message (str): Message to log.
        """
        if self._should_log():
            self.logger.info(f"[{module}] PROGRESS: {message}")

    # ==================== TensorBoard Logging Methods ====================

    def log_metrics(self, metrics: dict, step: int, model_name: str = ""):
        """
        Log evaluation metrics to TensorBoard.

        This method logs metrics to TensorBoard, handling various metric value types
        including scalars, arrays, and nested dictionaries. NaN values are skipped.

        Args:
            metrics (dict): Dictionary of metrics to log. Values may be scalars,
                arrays, or nested dictionaries.
            step (int): The current step (e.g., epoch, batch, or experiment ID).
            model_name (str): Optional prefix for metric names to group them in
                TensorBoard. Defaults to empty string.

        Returns:
            None: Logs metrics to TensorBoard if TensorBoard logging is enabled.
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
            figure: Matplotlib figure object to log.
            tag (str): Tag for the figure in TensorBoard.
            step (int): Step number for this figure.

        Returns:
            None: Logs figure to TensorBoard if TensorBoard logging is enabled.
        """
        if not self._should_log_tensorboard():
            return

        buf = io.BytesIO()
        figure.savefig(buf, format="png")
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue())
        image = tf.expand_dims(image, 0)
        with self.tf_writer.as_default():
            tf.summary.image(tag, image, step=step)
        self.tf_writer.flush()

    def log_image_file(self, image_path: str, tag: str, step: int):
        """
        Log an image from disk to TensorBoard.

        Args:
            image_path (str): Path to the image file to log.
            tag (str): Tag for the image in TensorBoard.
            step (int): Step number for this image.

        Returns:
            None: Logs image to TensorBoard if TensorBoard logging is enabled.
                Silently skips logging if file cannot be read.
        """
        if not self._should_log_tensorboard():
            return

        try:
            with open(image_path, "rb") as f:
                data = f.read()
            image = tf.image.decode_image(data)
            image = tf.expand_dims(image, 0)
            with self.tf_writer.as_default():
                tf.summary.image(tag, image, step=step)
            self.tf_writer.flush()
        except Exception as e:
            # Silently skip failed image logging (TensorBoard-only logging)
            pass

    def log_training_progress(
        self,
        model_name: str,
        epoch: int,
        loss: float,
        val_loss: Optional[float] = None,
        step: Optional[int] = None,
    ):
        """
        Log training progress for real-time monitoring.

        This method logs training and validation losses to TensorBoard for
        real-time monitoring of model training progress.

        Args:
            model_name (str): Name of the model being trained.
            epoch (int): Current epoch number.
            loss (float): Training loss value.
            val_loss (Optional[float]): Validation loss value. If None, only
                training loss is logged.
            step (Optional[int]): Global step for TensorBoard. If None, uses
                epoch as the step.

        Returns:
            None: Logs training progress to TensorBoard if TensorBoard logging
                is enabled.
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

    def log_hparams(self, hparams: dict, metrics: dict):
        """
        Log hyperparameters and metrics for TensorBoard's HParams dashboard.

        This method logs hyperparameters and their resulting metrics to TensorBoard's
        HParams dashboard for hyperparameter comparison and analysis.

        Args:
            hparams (dict): Dictionary of hyperparameters used for the run.
                Example: {'learning_rate': 0.1, 'model': 'XGBoost'}.
                Only scalar values (str, bool, int, float) are logged.
            metrics (dict): Dictionary of final metrics for this run.
                Example: {'mae': 12.3}. Only scalar values are logged.

        Returns:
            None: Logs hyperparameters and metrics to TensorBoard if TensorBoard
                logging is enabled.
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
            tag (str): Tag for the text in TensorBoard.
            text (str): Text content to log.
            step (int): Step number for this text.

        Returns:
            None: Logs text to TensorBoard if TensorBoard logging is enabled.
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
            tag (str): Tag for the scalar in TensorBoard.
            value (float): Scalar value to log.
            step (int): Step number for this scalar.

        Returns:
            None: Logs scalar to TensorBoard if TensorBoard logging is enabled.
        """
        if not self._should_log_tensorboard():
            return

        with self.tf_writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.tf_writer.flush()

    def close(self):
        """
        Flush and close all logger resources.

        This method flushes all log handlers and closes the TensorBoard writer.
        It can be called independently without using the context manager.

        Returns:
            None: Flushes and closes all logging resources.
        """
        # Flush all handlers in the standard logger
        for handler in self.logger.handlers:
            handler.flush()

        # Flush and close TensorBoard writer
        if self.tf_writer is not None:
            try:
                self.tf_writer.flush()
            except Exception:
                pass
            try:
                self.tf_writer.close()
            except Exception:
                pass
