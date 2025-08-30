import pandas as pd
import numpy as np
from typing import Dict, Any, Union
    
"""
Calculates Mean Absolute Scaled Error.
"""
class MASE:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes the MASE.
        Requires 'y_train' and optionally 'seasonal_period' in kwargs.
        """
        y_train = kwargs.get('y_train')
        seasonal_period = kwargs.get('seasonal_period', 1)

        if y_train is None:
            raise ValueError("y_train must be provided in kwargs for MASE calculation.")
        
        y_tr_eval = np.asarray(y_train)

        if y_true.ndim == 2: # Multivariate
            mase_scores = []
            for i in range(y_true.shape[1]):
                y_t_series = y_true[:, i]
                y_p_series = y_pred[:, i]
                y_tr_series = y_tr_eval[:, i] if y_tr_eval.ndim == 2 else y_tr_eval
                
                if len(y_tr_series) <= seasonal_period:
                    mase_scores.append(np.nan)
                    continue
                
                forecast_errors = np.abs(y_t_series - y_p_series)
                naive_errors = np.abs(y_tr_series[seasonal_period:] - y_tr_series[:-seasonal_period])
                scaler = np.mean(naive_errors)
                # Add small epsilon to prevent division by very small numbers
                epsilon = 1e-10
                mase_scores.append(np.mean(forecast_errors) / (scaler + epsilon) if scaler >= 0 else np.nan)
            return np.array(mase_scores)

        # Univariate
        forecast_errors = np.abs(y_true - y_pred)
        if len(y_tr_eval) <= seasonal_period: return np.nan
        naive_errors = np.abs(y_tr_eval[seasonal_period:] - y_tr_eval[:-seasonal_period])
        scaler = np.mean(naive_errors)
        # Add small epsilon to prevent division by very small numbers
        epsilon = 1e-10
        return np.mean(forecast_errors) / (scaler + epsilon) if scaler >= 0 else np.nan
