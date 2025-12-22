import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
from ..metrics.metrics import compute_and_smooth_power_spectrum

def plot_3D_attractor(context, prediction, ground_truth=None, lim_gen=2000, lim_pse=500, smoothing_sigma=2.0):
    """
    Plot 3D attractor with time series and power spectrum
    
    Args:
        context: Context data with shape (context_len, 3+)
        prediction: Model prediction with shape (pred_len, 3+)
        ground_truth: Optional ground truth data with shape (seq_len, 3+)
        lim_gen: Limit for time series plotting
        lim_pse: Limit for power spectrum plotting
        smoothing_sigma: Sigma for power spectrum smoothing
    
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if ground_truth is not None:
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = ground_truth.detach().cpu().numpy()
    if not isinstance(context, np.ndarray):
        context = context.detach().cpu().numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.detach().cpu().numpy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    # Set font sizes to match plotting.py
    plt.rcParams.update({'font.size': 14})
    
    CL_length = context.shape[0]
    
    # Create figure with width ratios
    fig = plt.figure(figsize=(18, 4))
    
    # Define grid with width ratios
    width_ratios = [1.5, 2.5, 1.2]
    sum_ratio = sum(width_ratios)
    widths = [r / sum_ratio for r in width_ratios]
    
    # Calculate positions (left edges)
    left_positions = [0.0]
    for i in range(2):
        left_positions.append(left_positions[-1] + widths[i])
    
    # 3D plot in first column - spans all rows
    ax3d = fig.add_axes([left_positions[0], 0.1, widths[0], 0.8], projection='3d')
    
    # Plot ground truth first if provided
    if ground_truth is not None:
        ax3d.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                  label='Ground truth', linewidth=2, color='#2C3E50', alpha=0.5)
    
    # Plot context
    ax3d.plot(context[:, 0], context[:, 1], context[:, 2],
              label='Context', linewidth=2, color='#2C3E50', alpha=0.9)
    
    # Plot prediction
    ax3d.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2],
              label='DynaMix', linewidth=2, color='#FF4242', alpha=0.9)
    
    ax3d.set_xlabel('X', fontsize=16)
    ax3d.set_ylabel('Y', fontsize=16)
    ax3d.set_zlabel('Z', fontsize=16)
    # Place legend above the plot
    ax3d.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.3, 1.25))
    ax3d.tick_params(axis='both', which='major', labelsize=12)
    
    # Time series plots (middle column)
    labels = ['X', 'Y', 'Z']
    height = 0.25
    for i in range(3):
        y_pos = (2-i) * (height + 0.15)  # Top to bottom positioning
        ax_ts = fig.add_axes([left_positions[1] + 0.02, y_pos, widths[1] - 0.05, height])
        
        # Plot ground truth with offset first if provided
        if ground_truth is not None:
            ax_ts.plot(np.arange(CL_length, CL_length + lim_gen), 
                       ground_truth[:lim_gen, i], 
                       label='Ground truth', linewidth=2.5, color='#2C3E50', alpha=0.5)
        
        # Plot context
        ax_ts.plot(np.arange(1, CL_length + 1), 
                   context[:, i], 
                   label='Context', linewidth=2.5, color='#2C3E50', alpha=0.9)
        
        # Plot prediction with offset
        ax_ts.plot(np.arange(CL_length, CL_length + lim_gen),
                   prediction[:lim_gen, i], 
                   label='DynaMix', linewidth=2.5, color='#FF4242', alpha=0.9)
        
        ax_ts.set_ylabel(labels[i], fontsize=16)
        if i == 2:
            ax_ts.set_xlabel('Time', fontsize=16)
        ax_ts.set_yticks([-2, 0, 2])
        ax_ts.grid(True, alpha=0.3)
        ax_ts.tick_params(axis='both', which='major', labelsize=12)
        
        #if i == 0:
        #    ax_ts.legend(fontsize=12)
    
    # Power spectrum plots (right column)
    for i in range(3):
        y_pos = (2-i) * (height + 0.15)
        ax_ps = fig.add_axes([left_positions[2] + 0.02, y_pos, widths[2] - 0.05, height])
        
        # Calculate and plot ground truth power spectrum first if provided
        if ground_truth is not None:
            gt_data = ground_truth[:, i].flatten() if isinstance(ground_truth[:, i], np.ndarray) else ground_truth[:, i].detach().cpu().numpy().flatten()
            ps = compute_and_smooth_power_spectrum(gt_data, smoothing_sigma)
            ax_ps.plot(ps[:lim_pse], label='Ground truth', linewidth=2.5, color='#2C3E50', alpha=0.5)
        
        # Calculate power spectrum for prediction
        pred_data = prediction[:, i].flatten() if isinstance(prediction[:, i], np.ndarray) else prediction[:, i].detach().cpu().numpy().flatten()
        ps_gen = compute_and_smooth_power_spectrum(pred_data, smoothing_sigma)
        
        # Plot prediction power spectrum
        ax_ps.plot(ps_gen[:lim_pse], label='DynaMix', linewidth=2.5, color='#FF4242', alpha=0.9)
        
        if i == 2:
            ax_ps.set_xlabel('Frequency', fontsize=16)
        
        ax_ps.set_yscale('log')
        ax_ps.grid(True, alpha=0.3)
        ax_ps.tick_params(axis='both', which='major', labelsize=12)
        
        #if i == 0:
        #    ax_ps.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_TS_forecast(context, prediction, ground_truth=None, lim=1000):
    """
    Plot time series forecast with context
    
    Args:
        context: Context data with shape (context_len, 1+)
        prediction: Model prediction with shape (pred_len, 1+)
        ground_truth: Optional ground truth data for future
        lim: Limit for forecasting horizon
    
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if ground_truth is not None:
        if not isinstance(ground_truth, np.ndarray):
            ground_truth = ground_truth.detach().cpu().numpy()
    if not isinstance(context, np.ndarray):
        context = context.detach().cpu().numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.detach().cpu().numpy()

    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Calculate time axis
    context_time = np.arange(-context.shape[0], 0)
    forecast_time = np.arange(0, lim)
    
    # Create figure
    fig = plt.figure(figsize=(14, 4))
    
    # Plot ground truth first if provided
    if ground_truth is not None:
        plt.plot(
            forecast_time, ground_truth[:lim, 0], 
            color='#2C3E50', linewidth=2.5, alpha=0.5, 
            label='Ground Truth'
        )
    
    # Plot context
    plt.plot(
        context_time, context[:, 0], 
        color='#2C3E50', linewidth=2.5, alpha=0.9, 
        label='Context'
    )
    
    # Plot prediction
    plt.plot(
        forecast_time, prediction[:lim, 0], 
        color='#FF4242', linewidth=2.5, alpha=0.9, 
        label='DynaMix'
    )
    
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    return fig

def plot_2D_attractor(context, prediction):
    """
    Plot 2D attractor with context and prediction
    
    Args:
        context: Context data with shape (context_len, 2+)
        prediction: Model prediction with shape (pred_len, 2+)
    
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    if not isinstance(context, np.ndarray):
        context = context.detach().cpu().numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.detach().cpu().numpy()

    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    # Set font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Create figure
    fig = plt.figure(figsize=(6, 6))
    
    # Plot context
    plt.plot(
        context[:, 0], context[:, 1],
        linewidth=4, color='#2C3E50', alpha=0.9, 
        label='Context'
    )
    
    # Plot prediction
    plt.plot(
        prediction[:, 0], prediction[:, 1],
        linewidth=4, color='#FF4242', alpha=0.9, 
        label='DynaMix'
    )
    
    # Configure plot appearance
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.xticks([-1, 0, 1, 2])
    plt.yticks([-1, 0, 1, 2, 3])
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    
    return fig
