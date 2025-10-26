"""
Trainer class for model training and evaluation.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from ..utils.tf_logger import get_tf_logger
from ..models.base_model import BaseModel
from .visualizer import Visualizer
import os


class Trainer: #update the trainer class to a folder for each model type
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None, logs_dir: str):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
            logs_dir: Directory for storing log files
        """
        self.config = config or {}
        if config_file:
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))
        
        # Initialize TensorBoard logger with config (sets up TensorBoard writer)
        tf_logs_dir = self.config.get('log_dir', self.config.get('logs_dir', logs_dir))
        self.tf_logger = get_tf_logger(tf_logs_dir)
        
    def train(self, model: BaseModel, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> BaseModel:
        """
        Train the model on the provided data.
        
        Args:
            model: Model instance to train
            X: Training features
            y: Target values
            
        Returns:
            Trained model instance
        """
        try:
            # Train the model
            model.train(X, y)
            
            return model
            
        except Exception as e:
            raise
            
    def evaluate(self, model: BaseModel, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.
        
        Args:
            model: Trained model instance
            X: Evaluation features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = model.evaluate(X, y)
            # Log metrics to TensorBoard (single step evaluation)
            try:
                self.tf_logger.log_metrics(metrics=metrics, step=1, model_name=model.__class__.__name__)
            except Exception as logging_error:
                # Fail-fast preference: surface error but do not hide evaluation result
                pass

            # Optionally create and log a forecast plot
            try:
                y_true_arr, y_pred_arr = model.get_last_eval_true_pred()
                if y_true_arr is not None and y_pred_arr is not None:
                    visualizer = Visualizer(self.config)
                    # Build save path if output_dir configured
                    output_dir = self.config.get('output_dir', 'outputs')
                    os.makedirs(output_dir, exist_ok=True)
                    plot_path = os.path.join(output_dir, f"forecast_{model.__class__.__name__}.png")
                    import matplotlib.pyplot as plt
                    plt.ioff()
                    visualizer.plot_predictions(y_true=y_true_arr, y_pred=y_pred_arr,
                                                title=f"Forecast - {model.__class__.__name__}",
                                                save_path=plot_path)
                    fig = plt.gcf()
                    self.tf_logger.log_figure(fig, tag=f"{model.__class__.__name__}/forecast", step=1)
                    plt.close(fig)
            except Exception as viz_error:
                pass
            return metrics
            
        except Exception as e:
            raise
            
    def train_and_evaluate(self, model: BaseModel, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train and evaluate the model on the provided data.
        
        Args:
            model: Model instance to train
            X: Training features
            y: Target values (used for both training and evaluation)
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        try:
            # Train the model
            trained_model = self.train(model, X, y)
            
            # Evaluate the model
            metrics = self.evaluate(trained_model, X, y)
            
            return trained_model, metrics
            
        except Exception as e:
            raise