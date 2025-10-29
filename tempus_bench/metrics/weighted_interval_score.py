"""
Calculates the Weighted Interval Score (WIS) for deciles.

The Weighted Interval Score is a proper scoring rule for probabilistic forecasts
that evaluates prediction intervals at multiple confidence levels. It combines
sharpness (interval width) and calibration (coverage) into a single score.

Mathematical formula:
WIS_A = Σ_{α∈A} α * IS_α([l_α, u_α], y)

Where:
- IS_α = (u_α - l_α) + (2/α) * dist(y, [l_α, u_α])
- l_α = F⁻¹(α/2) and u_α = F⁻¹(1 - α/2) are the quantiles from samples
- dist(y, [l_α, u_α]) = max(0, max(l_α - y, y - u_α)) is the distance penalty
- α is the significance level for each prediction interval

The score rewards narrow intervals (sharpness) and penalizes when the true value
falls outside the prediction interval (calibration).
"""
import numpy as np


class WeightedIntervalScore:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        Computes Weighted Interval Score (WIS) for deciles.

        Args:
            y_true: True values, shape (n_timesteps, 1) or (n_timesteps, num_targets)
            y_pred: Prediction samples, shape (num_samples, forecast_horizon, num_targets)
            **kwargs: Must contain optional 'task_type'
                - task_type: Optional, defaults to 'stochastic'

        Returns:
            float: WIS score averaged across all timesteps and targets

        Raises:
            ValueError: If task_type is not 'stochastic'
        """
        task_type = kwargs.get('task_type')

        if task_type != 'stochastic':
            raise ValueError(f"WeightedIntervalScore can only be used with 'stochastic' task_type, got '{task_type}'.")

        WIS = np.zeros(y_true.shape)
        for a in np.linspace(0.0, 1.0, num=kwargs['num_quantiles']):
            l = np.quantile(y_pred, a/2, axis=0)      # Lower bound: F⁻¹(α/2)
            u = np.quantile(y_pred, 1 - a/2, axis=0)  # Upper bound: F⁻¹(1 - α/2)
            dist = np.maximum(0, np.maximum(l - y_true, y_true - u))
            IS_a = (u - l) + (2.0/a)*dist
            WIS += a*IS_a

        return np.mean(WIS)
