import os
import pandas as pd
import numpy as np
import io
from typing import Optional

# Configure TensorFlow threading before import
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
from datetime import datetime
from tensorboard.plugins.hparams import api as hp

"""
TensorBoard logging utilities for metrics, plots, and hyperparameters.
This logger only writes to TensorBoard files - no console output.
"""

class TFLogger:
    def __init__(self, tf_logs_dir: str, name: str = "TFLogger"):
        """
        Initialize TensorBoard logger with configuration.

        Args:
            tf_logs_dir: Directory to write TensorBoard log files (used directly)
            name: Name for the logger instance (for identification only)
        """
        self.name = name
        os.makedirs(tf_logs_dir, exist_ok=True)
        self.tf_logs_dir = tf_logs_dir
        self.tf_writer = tf.summary.create_file_writer(self.tf_logs_dir)

    def log_metrics(self, metrics, step, model_name=""):
        """
        Log evaluation metrics to TensorBoard.

        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): The current step (e.g., epoch, batch, or experiment ID).
            model_name (str, optional): A prefix for metric names to group them in TensorBoard.
        """
        group_prefix = f"{model_name}/" if model_name else ""
        with self.tf_writer.as_default():
            for metric_name, value in metrics.items():
                if value is None or (isinstance(value, (float, int)) and np.isnan(value)):
                    continue

                if isinstance(value, dict): # For nested results like quantile losses
                    for sub_metric, sub_value in value.items():
                        tag = f"{group_prefix}{metric_name}/{sub_metric}"
                        tf.summary.scalar(tag, sub_value, step=step)
                elif isinstance(value, (np.ndarray, list)):
                    # For arrays log each element and the mean
                    value_arr = np.asarray(value)
                    tf.summary.scalar(f"{group_prefix}{metric_name}/mean", np.nanmean(value_arr), step=step)
                    for i, elem in enumerate(value_arr):
                        tf.summary.scalar(f"{group_prefix}{metric_name}/series_{i}", elem, step=step)
                elif isinstance(value, (float, np.floating, int)):
                    tf.summary.scalar(f"{group_prefix}{metric_name}", value, step=step)
                else:
                    # Skip unsupported metric types silently (TensorBoard-only logging)
                    pass
        self.tf_writer.flush()

    def log_figure(self, figure, tag: str, step: int):
        """
        Log a Matplotlib figure to TensorBoard.
        """
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        with self.tf_writer.as_default():
            tf.summary.image(tag, image, step=step)
        self.tf_writer.flush()

    def log_image_file(self, image_path: str, tag: str, step: int):
        """
        Log an image from disk to TensorBoard.
        """
        try:
            with open(image_path, 'rb') as f:
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
        with self.tf_writer.as_default():
            # Sanitize hparams for TensorBoard (it has specific type requirements)
            sanitized_hparams = {k: v for k, v in hparams.items() if isinstance(v, (str, bool, int, float))}
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
        with self.tf_writer.as_default():
            tf.summary.scalar(tag, value, step=step)
        self.tf_writer.flush()

    def close(self):
        """Closes the TensorBoard writer to ensure all data is written to disk."""
        self.tf_writer.close()


# Global TFLogger instance
_global_tf_logger = None

def get_tf_logger(tf_logs_dir: str) -> TFLogger:
    """
    Get or create the global TFLogger instance.
    
    Args:
        tf_logs_dir: Directory to write TensorBoard log files
        
    Returns:
        TFLogger: Global TFLogger instance
    """
    global _global_tf_logger
    if _global_tf_logger is None or _global_tf_logger.tf_logs_dir != tf_logs_dir:
        _global_tf_logger = TFLogger(tf_logs_dir)
    return _global_tf_logger
