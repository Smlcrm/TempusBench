import torch
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import random
from statsmodels.tsa.stattools import acf
from scipy.ndimage import gaussian_filter1d
from scipy import optimize
from scipy.optimize import curve_fit


class TimeSeriesProcessor:
    """
    Utility class for converting between numpy and torch.
    """
    @staticmethod
    def to_numpy(data):
        """Convert torch tensor to numpy array while preserving device and dtype info"""
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            dtype = data.dtype
            return data.detach().cpu().numpy(), is_torch, device, dtype
        return data, False, None, None
    
    @staticmethod
    def to_torch(data_np, is_torch, device=None, dtype=None):
        """Convert numpy array back to torch tensor if original was a tensor"""
        if is_torch:
            return torch.tensor(data_np, device=device, dtype=dtype)
        return data_np


class Embedding:
    """
    Class for embedding methods to transform time series to target dimension.
    """
    @staticmethod
    def estimate_TDM_tau(data, acorr_threshold=1/np.e):
        """
        Estimate tau using autocorrelation function with threshold method
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            acorr_threshold: Autocorrelation threshold

        Returns:
            Maximum estimated tau across all dimensions
        """
        # Convert to numpy
        data_np, _, _, _ = TimeSeriesProcessor.to_numpy(data)
        
        seq_length, n_dims = data_np.shape
        tau_vals = np.zeros(n_dims, dtype=int)
        
        for dim in range(n_dims):
            # Calculate autocorrelation
            autocorr_vals = acf(data_np[:, dim] - np.mean(data_np[:, dim]), nlags=seq_length//2)
            
            # Find first value below threshold (after lag 0)
            below_threshold = np.where(autocorr_vals[1:] < acorr_threshold)[0]
            if len(below_threshold) > 0:
                tau_vals[dim] = below_threshold[0] + 1  # +1 because skipping lag 0
            else:
                tau_vals[dim] = 1  # Default if no value below threshold
        
        return int(np.max(tau_vals))
    
    @staticmethod
    def estimate_pos_tau(data, max_lag=None, min_lag=None):
        """
        Estimate autocorrelation time for positional embedding
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            max_lag: Maximum lag to consider
            min_lag: Minimum lag to consider
            
        Returns:
            Maximum autocorrelation time across dimensions
        """
        data_np, _, _, _ = TimeSeriesProcessor.to_numpy(data)
        seq_length, n = data_np.shape
        
        if max_lag is None:
            max_lag = seq_length - 1
        if min_lag is None:
            min_lag = seq_length // 10
            
        tau_vals = np.zeros(n, dtype=int)
        
        for dim in range(n):
            ts = data_np[:, dim] if not isinstance(data, torch.Tensor) else data[:, dim].cpu().numpy()
            autocorr_vals = acf(ts - np.mean(ts), nlags=max_lag)
            
            # Determine max autocorrelation with tau>tau_min
            peaks, _ = find_peaks(autocorr_vals)
            valid_peaks = [i for i in peaks if i > min_lag and i < len(autocorr_vals)]
            if valid_peaks:
                peak_values = autocorr_vals[valid_peaks]
                max_peak_idx = np.argmax(peak_values)
                tau_vals[dim] = valid_peaks[max_peak_idx]
            else:
                start_idx = min_lag + 1
                segment = autocorr_vals[start_idx:]
                tau_vals[dim] = start_idx + int(np.argmax(segment))
        
        return np.max(tau_vals)
    
    @staticmethod
    def delay_embedding(data, model_dim, tau=None):
        """
        Standard delay embedding with optimal tau
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            model_dim: Target dimension
            tau: Time delay (if None, estimated from autocorrelation)
            
        Returns:
            Delay embedded data of shape (shortened_length, model_dim)
        """
        seq_length, N_data = data.shape
        needed_dims = model_dim - N_data
        
        if needed_dims <= 0:
            return data
        
        processed_data = data.clone()
        
        # Estimate tau if not provided
        if tau is None:
            tau = Embedding.estimate_TDM_tau(processed_data)
        
        # Select the last column for embedding
        ts = processed_data[:, -1].clone()
        
        # Calculate starting index
        start_idx = needed_dims * tau
        
        # Handle case where start_idx is too large
        if start_idx >= seq_length:
            tau = max(1, seq_length // (needed_dims + 1))
            start_idx = needed_dims * tau
        
        # Create shortened data
        shortened_data = processed_data[start_idx:].clone()
        result = shortened_data
        
        # Add delayed versions
        for i in range(1, needed_dims + 1):
            delayed = ts[start_idx - i * tau:seq_length - i * tau].unsqueeze(1)
            result = torch.cat([result, delayed], dim=1)
        
        return result
    
    @staticmethod
    def delay_embedding_random(data, model_dim, upper_tau=10, lower_tau=3):
        """
        Random delay embedding with random tau values
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            model_dim: Target dimension
            upper_tau: Upper bound for random tau values
            lower_tau: Lower bound for random tau values
            
        Returns:
            Random delay embedded data
        """
        seq_length, N_data = data.shape
        needed_dims = model_dim - N_data
        
        if needed_dims <= 0:
            return data
        
        processed_data = data.clone()
        
        # Generate random tau values
        taus = [random.randint(lower_tau, upper_tau) for _ in range(needed_dims)]
        max_tau = max(taus)
        
        # Select the first column for embedding
        ts = processed_data[:, 0].clone()
        
        # Create shortened data
        result = processed_data[max_tau:].clone()
        
        # Add delayed versions
        for i in range(needed_dims):
            delayed = ts[max_tau - taus[i]:seq_length - taus[i]].unsqueeze(1)
            result = torch.cat([result, delayed], dim=1)
        
        return result
    
    @staticmethod
    def zero_embedding(data, model_dim):
        """
        Zero embedding: appends zeros to reach model dimensions
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            model_dim: Target dimension
            
        Returns:
            Tensor with zeros appended to reach model_dim
        """
        seq_length, N_data = data.shape
        needed_dims = model_dim - N_data
        
        if needed_dims > 0:
            zeros = torch.zeros(seq_length, needed_dims, device=data.device, dtype=data.dtype)
            data = torch.cat([data, zeros], dim=1)
        
        return data
    
    @staticmethod
    def positional_embedding(data, model_dim, tau=None):
        """
        Positional embedding: adds sinusoidal signals based on autocorrelation time
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            model_dim: Target dimension
            tau: Optional fixed value for tau. If None, estimated from data.
            
        Returns:
            Data with positional embeddings added
        """
        seq_length, N_data = data.shape
        needed_dims = model_dim - N_data
        
        if needed_dims <= 0:
            return data
            
        if needed_dims != 1:
            shifts = torch.linspace(0, np.pi/2, needed_dims, device=data.device)
        else:
            shifts = torch.tensor([0.0], device=data.device)
            
        tau_val = tau if tau is not None else Embedding.estimate_pos_tau(data)
        t = torch.arange(1, seq_length + 1, dtype=data.dtype, device=data.device)
        
        result = data.clone()
        for shift in shifts:
            pos_feature = torch.sin(2 * np.pi / tau_val * t + shift).unsqueeze(1)
            result = torch.cat([result, pos_feature], dim=1)
        
        return result
    
    @staticmethod
    def apply_embedding(data, model_dim, method="pos_embedding", **kwargs):
        """
        Apply selected embedding method to the data
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            model_dim: Target dimension
            method: Embedding method ('pos_embedding', 'zero_embedding', 
                    'delay_embedding', or 'delay_embedding_random')
            **kwargs: Additional parameters to pass to the specific embedding method
            
        Returns:
            Embedded data
        """
        if method == "pos_embedding":
            return Embedding.positional_embedding(data, model_dim, **kwargs)
        elif method == "zero_embedding":
            return Embedding.zero_embedding(data, model_dim)
        elif method == "delay_embedding":
            return Embedding.delay_embedding(data, model_dim, **kwargs)
        elif method == "delay_embedding_random":
            return Embedding.delay_embedding_random(data, model_dim, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding method: {method}")


class BoxCoxTransformer:
    """
    Applies Box-Cox transformation to data for variance stabilization.
    """
    def __init__(self, lambda_range=(-2, 2)):
        """
        Initialize BoxCoxTransformer.
        
        Args:
            lambda_range: Range for lambda parameter search
        """
        self.lambda_range = lambda_range
        self.params = None
    
    @staticmethod
    def transform(data, lambda_range=(-2, 2)):
        """
        Apply Box-Cox transformation to data for stabilization
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            lambda_range: Range for lambda parameter search
            
        Returns:
            Transformed data and parameters for inverse transformation
        """
        # Convert to numpy
        data_np, is_torch, device, dtype = TimeSeriesProcessor.to_numpy(data)
        
        seq_length, n_dims = data_np.shape
        transformed_data = np.zeros_like(data_np)
        box_cox_params = []
        
        for dim in range(n_dims):
            # Add constant to ensure positivity
            if np.min(data_np[:, dim]) <= 0:
                offset = abs(np.min(data_np[:, dim])) + 1.2
                data_shifted = data_np[:, dim] + offset
            else:
                offset = 1.2
                data_shifted = data_np[:, dim] + offset
            
            try:
                # Find optimal lambda for Box-Cox transformation
                transformed, lambda_param = stats.boxcox(data_shifted)
                
                # Limit lambda to a reasonable range to prevent numerical issues
                lambda_param = max(min(lambda_param, 2.0), -2.0)
                
                # Recalculate transformation with bounded lambda for consistency
                if abs(lambda_param) < 1e-8:
                    # For lambda near zero, use logarithmic transformation
                    transformed = np.log(data_shifted)
                else:
                    transformed = (data_shifted ** lambda_param - 1) / lambda_param
                
                # Store transformed data and parameters
                transformed_data[:, dim] = transformed
            except:
                # If transformation fails, just use the original data
                transformed_data[:, dim] = data_np[:, dim]
                lambda_param = 1.0  # Identity transform
                
            box_cox_params.append((lambda_param, offset))
        
        # Convert back to torch if needed
        return TimeSeriesProcessor.to_torch(transformed_data, is_torch, device, dtype), box_cox_params
    
    @staticmethod
    def inverse_transform(data, box_cox_params):
        """
        Apply inverse Box-Cox transformation
        
        Args:
            data: Transformed data tensor
            box_cox_params: Parameters from Box-Cox transformation
            
        Returns:
            Original scale data
        """
        # Convert to numpy for computation
        data_np, is_torch, device, dtype = TimeSeriesProcessor.to_numpy(data)
        
        seq_length, n_dims = data_np.shape
        inverse_data = np.zeros_like(data_np)
        
        for dim in range(min(n_dims, len(box_cox_params))):
            lambda_param, offset = box_cox_params[dim]
            
            # Apply inverse transformation
            if abs(lambda_param) < 1e-8:
                # For lambda near zero, the transformation is logarithmic
                inverse_data[:, dim] = np.exp(data_np[:, dim]) - offset
            elif abs(lambda_param - 1.0) < 1e-8:
                # For lambda=1 (identity transform), just subtract offset
                inverse_data[:, dim] = data_np[:, dim] - offset
            else:
                # For other lambda values
                base = lambda_param * data_np[:, dim] + 1
                
                # Simple clipping approach to ensure base is positive
                # This avoids complex numbers while preserving most data characteristics
                base = np.maximum(base, 1e-10)
                
                # Apply power transformation
                result = base ** (1/lambda_param)
                inverse_data[:, dim] = result - offset
        
        # Convert back to torch if needed
        return TimeSeriesProcessor.to_torch(inverse_data, is_torch, device, dtype)


class Detrending:
    """
    Applies detrending model to time series data.
    """
    @staticmethod
    def trend_model(t, params):
        """
        Model for detrending
        
        Args:
            t: Time points
            params: Model parameters [a, b, c]
            
        Returns:
            Model values
        """
        a, b, c = params
        return a * (t ** b) + c
    
    @staticmethod
    def fit_objective(params, data):
        """
        Objective function for trend model fitting
        
        Args:
            params: Model parameters
            data: Data to fit
            
        Returns:
            Sum of squared errors
        """
        t = np.arange(1, len(data) + 1)
        predicted = Detrending.trend_model(t, params)
        return np.sum((data - predicted) ** 2)
    
    @staticmethod
    def apply_detrending(data):
        """
        Apply trend model to data
        
        Args:
            data: Input data tensor of shape (seq_length, N)
            
        Returns:
            Detrended data and parameters for inverse transformation
        """
        # Convert to numpy
        data_np, is_torch, device, dtype = TimeSeriesProcessor.to_numpy(data)
        
        seq_length, n_dims = data_np.shape
        detrended_data = np.zeros_like(data_np)
        detrending_params = []
        
        for dim in range(n_dims):
            # Define the objective function for this dimension
            objective = lambda params: Detrending.fit_objective(params, data_np[:, dim])
            
            # Initial parameter guess
            initial_params = [0.0, 1.0, data_np[0,dim]]
            
            # Bounds for parameters
            bounds = [(None, None), (0.0, 3.0), (None, None)]
            
            # Optimize
            result = optimize.minimize(
                objective, 
                initial_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={
                    'maxiter': 5000,
                    'gtol': 1e-6,
                    'maxfun': 1500,
                    'maxcor': 10
                }
            )
            optimal_params = np.round(result.x, 10)
            
            # Calculate trend and detrend the data
            t = np.arange(1, seq_length + 1)
            trend = Detrending.trend_model(t, optimal_params)
            detrended_data[:, dim] = data_np[:, dim] - trend
            
            # Store parameters for inverse transformation
            detrending_params.append(optimal_params)
        
        # Convert back to torch if needed
        return TimeSeriesProcessor.to_torch(detrended_data, is_torch, device, dtype), detrending_params
    
    @staticmethod
    def apply_detrending_inverse(context, data, detrending_params):
        """
        Apply inverse detrending to forecasted data
        
        Args:
            context: Original context data
            data: Forecasted data
            detrending_params: Parameters from detrending
            
        Returns:
            Forecasted data with trend restored
        """
        # Convert to numpy for computation
        data_np, is_torch, device, dtype = TimeSeriesProcessor.to_numpy(data)
        context_np, _, _, _ = TimeSeriesProcessor.to_numpy(context)
        
        # Get dimensions
        forecast_length, n_dims = data_np.shape
        context_length = len(context_np)
        
        # Create time points for the forecast horizon
        t = np.arange(context_length + 1, context_length + forecast_length + 1)
        
        # Add trend back to each dimension
        for dim in range(min(n_dims, len(detrending_params))):
            params = detrending_params[dim]
            trend = Detrending.trend_model(t, params)
            data_np[:, dim] = data_np[:, dim] + trend
        
        # Convert back to torch if needed
        return TimeSeriesProcessor.to_torch(data_np, is_torch, device, dtype)


def estimate_initial_condition(initial_x, context_embedded):
    """
    Estimate full initial condition from partial observation
    
    Args:
        initial_x: Partial initial condition of shape (N_partial,)
        context_embedded: Context data of shape (seq_length, N)
        
    Returns:
        Complete initial condition of shape (N,)
    """
    T, N = context_embedded.shape
    N_partial = initial_x.shape[0]
    
    assert N_partial <= N, "Initial condition dimension must be <= embedding dimension"
    
    # Find timestep with closest match to initial condition in first N_partial dimensions
    distances = torch.zeros(T, device=initial_x.device)
    for t in range(T):
        distances[t] = torch.sum((context_embedded[t, :N_partial] - initial_x) ** 2)
        
    closest_t = torch.argmin(distances)
    
    # Combine initial condition with closest matching state
    return torch.cat([initial_x, context_embedded[closest_t, N_partial:]])


# Legacy functions for backward compatibility
def tau_delay_embedding(data, threshold=0.368):
    return Embedding.estimate_TDM_tau(data, threshold)

def tau_pos_embedding(data, max_lag=None, min_lag=None):
    return Embedding.estimate_pos_tau(data, max_lag, min_lag)

def delay_embedding(data, model_dim, tau=None):
    return Embedding.delay_embedding(data, model_dim, tau)

def delay_embedding_random(data, model_dim, upper_tau=10, lower_tau=3):
    return Embedding.delay_embedding_random(data, model_dim, upper_tau, lower_tau)

def zero_embedding(data, model_dim):
    return Embedding.zero_embedding(data, model_dim)

def pos_embedding(data, model_dim):
    return Embedding.positional_embedding(data, model_dim)

def data_preprocessing(data, model_dim, preprocessing_method="pos_embedding", **kwargs):
    return Embedding.apply_embedding(data, model_dim, preprocessing_method, **kwargs)

def apply_box_cox(data, lambda_range=(-2, 2)):
    return BoxCoxTransformer.transform(data, lambda_range)

def apply_box_cox_inverse(data, box_cox_params):
    return BoxCoxTransformer.inverse_transform(data, box_cox_params)

def apply_detrending(data):
    return Detrending.apply_detrending(data)

def apply_detrending_inverse(context, data, detrending_params):
    return Detrending.apply_detrending_inverse(context, data, detrending_params)
