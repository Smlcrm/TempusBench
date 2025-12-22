"""
Plotting utilities for dynamical systems.
"""

import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import os
import math

def plot_trajectories(real_traj, pred_traj, save_dir=None, name_prefix="", plot_length=1000):
    """
    Plot trajectories with flexible handling of different dimensions.
    
    Cases:
    - For N=3: 3D attractor on left and time series on right
    - For N=2: 2D phase plot on left and time series on right
    - For N=1 or N>3: Only time series plots (up to 10 dimensions)
    
    Args:
        real_traj: The ground truth trajectory [T, N]
        pred_traj: The predicted trajectory [T, N]
        save_dir: Directory to save plots
        name_prefix: Prefix for the saved plot names
        plot_length: Number of timesteps to plot in the time series
    """
    # Convert tensors to numpy if needed
    if isinstance(real_traj, torch.Tensor):
        real_traj = real_traj.cpu().numpy()
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu().numpy()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    
    # Get number of dimensions
    n_dims = real_traj.shape[1]
    time = np.arange(min(len(real_traj), plot_length))
    dim_labels = [f'Dimension {i+1}' for i in range(n_dims)]
    
    # Set larger font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Handle different dimensionality cases
    if n_dims == 3:
        # Case 1: 3D data - create 3D plot + time series
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], wspace=0.4)
        
        # 3D plot for the attractor on the left spanning all rows
        ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
        ax_3d.plot(real_traj[:, 0], real_traj[:, 1], real_traj[:, 2], 
                  color='#2C3E50', label='Ground Truth', linewidth=2.5, alpha=0.5)
        ax_3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
                  color='#FF4242', label='DynaMix', linewidth=2.5, alpha=0.9)
        ax_3d.set_xlabel('Dimension 1', fontsize=16)
        ax_3d.set_ylabel('Dimension 2', fontsize=16)
        ax_3d.set_zlabel('Dimension 3', fontsize=16)
        ax_3d.legend(fontsize=14)
        ax_3d.tick_params(axis='both', which='major', labelsize=12)
        
        # Individual time series plots on the right
        for i in range(n_dims):
            ax_ts = fig.add_subplot(gs[i, 1])
            ax_ts.plot(time, real_traj[:plot_length, i], color='#2C3E50', label='Ground Truth', linewidth=2.5, alpha=0.5)
            ax_ts.plot(time, pred_traj[:plot_length, i], color='#FF4242', label='DynaMix', linewidth=2.5, alpha=0.9)
            ax_ts.set_ylabel(dim_labels[i], fontsize=16)
            ax_ts.grid(True, alpha=0.3)
            ax_ts.tick_params(axis='both', which='major', labelsize=12)
            
            # Add legend to first plot, x-label to bottom plot
            if i == 0:
                ax_ts.legend(fontsize=12)
            if i == n_dims - 1:
                ax_ts.set_xlabel('Time', fontsize=16)
                
    elif n_dims == 2:
        # Case 2: 2D data - create 2D phase plot
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.4)
        
        # 2D phase plot on the left spanning all rows
        ax_2d = fig.add_subplot(gs[:, 0])
        ax_2d.plot(real_traj[:, 0], real_traj[:, 1], color='#2C3E50', label='Ground Truth', 
                  linewidth=2.5, alpha=0.5)
        ax_2d.plot(pred_traj[:, 0], pred_traj[:, 1], color='#FF4242', label='DynaMix', 
                  linewidth=2.5, alpha=0.9)
        ax_2d.set_xlabel(dim_labels[0], fontsize=16)
        ax_2d.set_ylabel(dim_labels[1], fontsize=16)
        ax_2d.legend(fontsize=14)
        ax_2d.grid(True, alpha=0.3)
        ax_2d.tick_params(axis='both', which='major', labelsize=12)
        
        # Individual time series plots on the right
        for i in range(n_dims):
            ax_ts = fig.add_subplot(gs[i, 1])
            ax_ts.plot(time, real_traj[:plot_length, i], color='#2C3E50', label='Ground Truth', linewidth=2.5, alpha=0.5)
            ax_ts.plot(time, pred_traj[:plot_length, i], color='#FF4242', label='DynaMix', linewidth=2.5, alpha=0.9)
            ax_ts.set_ylabel(dim_labels[i], fontsize=16)
            ax_ts.grid(True, alpha=0.3)
            ax_ts.tick_params(axis='both', which='major', labelsize=12)
            
            # Add legend to first plot, x-label to bottom plot
            if i == 0:
                ax_ts.legend(fontsize=12)
            if i == n_dims - 1:
                ax_ts.set_xlabel('Time', fontsize=16)
                
    else:
        # Case 3: 1D or >3D - create only time series plots (limit to 10 dimensions)
        plot_dims = min(n_dims, 10)  # Limit to 10 dimensions
        rows = math.ceil(plot_dims / 2)  # 2 columns, calculate needed rows
        
        fig = plt.figure(figsize=(20, 4*rows))
        gs = fig.add_gridspec(rows, 2, wspace=0.3, hspace=0.4)
        
        for i in range(plot_dims):
            row, col = divmod(i, 2)
            ax_ts = fig.add_subplot(gs[row, col])
            ax_ts.plot(time, real_traj[:plot_length, i], color='#2C3E50', label='Ground Truth', linewidth=2.5, alpha=0.5)
            ax_ts.plot(time, pred_traj[:plot_length, i], color='#FF4242', label='DynaMix', linewidth=2.5, alpha=0.9)
            ax_ts.set_ylabel(dim_labels[i], fontsize=16)
            ax_ts.grid(True, alpha=0.3)
            ax_ts.tick_params(axis='both', which='major', labelsize=12)
            
            # Add legend to first plot
            if i == 0:
                ax_ts.legend(fontsize=12)
                
            # Add x-label to bottom plots
            if row == rows - 1:
                ax_ts.set_xlabel('Time', fontsize=16)
                
        # Add note if dimensions were limited
        if n_dims > 10:
            fig.suptitle(f'Showing first 10 of {n_dims} dimensions', fontsize=16)
    
    if save_dir:
        plot_path = Path(save_dir) / f"{name_prefix}trajectory_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_metrics(metrics_data, save_dir=None, name_prefix="", show=False):
    """
    Plot training metrics over time.
    
    Args:
        metrics_data: Dictionary containing metrics data
        save_dir: Directory to save plots
        name_prefix: Prefix for the saved plot names
        show: Whether to display the plots
    """
    # Extract metrics
    epochs = metrics_data.get('epochs', None)
    losses = metrics_data.get('losses', None)
    dstsp = metrics_data.get('dstsp', None)
    dh = metrics_data.get('dh', None)
    pe = metrics_data.get('pe', None)
    
    if epochs is None or not any([losses, dstsp, dh, pe]):
        print("No metrics data available to plot")
        return
    
    # Set larger font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Plot training loss
    if losses is not None:
        plt.figure(figsize=(10, 6))
        # Ensure x-axis matches the length of losses
        x_axis = np.arange(len(losses)) if epochs is None or len(epochs) != len(losses) else epochs
        plt.plot(x_axis, losses, color='#2C3E50', linewidth=2.5, alpha=0.9)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.title('Training Loss', fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.yscale('log')
        
        if save_dir:
            loss_path = Path(save_dir) / f"{name_prefix}loss.png"
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # Plot evaluation metrics
    metrics_to_plot = []
    if dstsp is not None and epochs is not None and len(dstsp) == len(epochs):
        metrics_to_plot.append(('Dstsp', dstsp))
    if dh is not None and epochs is not None and len(dh) == len(epochs):
        metrics_to_plot.append(('DH', dh))
    if pe is not None and epochs is not None and len(pe) == len(epochs):
        metrics_to_plot.append(('PE', pe))
    
    if metrics_to_plot and epochs:
        plt.figure(figsize=(12, 8))
        
        for i, (name, metric) in enumerate(metrics_to_plot):
            plt.subplot(len(metrics_to_plot), 1, i+1)
            plt.plot(epochs, metric, color='#2C3E50', linewidth=2.5, alpha=0.9)
            plt.ylabel(name, fontsize=16)
            if i == len(metrics_to_plot) - 1:
                plt.xlabel('Epoch', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        if save_dir:
            metrics_path = Path(save_dir) / f"{name_prefix}metrics.png"
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
