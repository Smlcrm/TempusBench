import numpy as np

class BaseMetric:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        pass

    def process_y_pred(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> np.ndarray:
        task_type = kwargs.get('task_type')
        S, T, M = y_pred.shape
        if y_true.shape != (T, M):
            raise ValueError(f"Shape mismatch: y_true has shape {y_true.shape}, but expected ({T}, {M}) to match y_pred (num_samples={S}, time_steps={T}, num_targets={M})")
        if len(y_pred.shape) > 2:
            raise ValueError("y_pred can't have more than 2 dimensions for deterministic evaluation")

        if task_type == 'stochastic':
            point_forecast_statistic = kwargs['point_forecast_statistic']
            if point_forecast_statistic == 'mean':
                y_ppred = np.mean(y_pred, axis=0)
            else:
                raise ValueError("RMSE can only handle point_forecast_statistic == 'mean' for stochastic evaluation.")
        else:
            y_ppred = y_pred
