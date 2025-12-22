"""
Main entry point for Zero-shot DSR training.
"""

import numpy as np
import torch
import torch.nn as nn
import argparse
import json
import os
from pathlib import Path

from ..model.dynamix import DynaMix, print_model_parameters
from ..utilities.dataset import Dataset
from .training import train_dynamix

def setup_gpu(args):
    """Configure GPU settings based on provided arguments."""
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            args.device = 'cpu'
            return torch.device('cpu')
        
        # Check if requested GPU ID is valid
        num_gpus = torch.cuda.device_count()
        if args.gpu_id >= num_gpus:
            print(f"WARNING: GPU {args.gpu_id} requested but only {num_gpus} GPUs available. Using GPU 0.")
            args.gpu_id = 0
            
        # Set device to specific GPU
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_dtype(torch.float32)
            
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU for computation")
        
    return device

def load_settings(settings_file):
    """Load settings from JSON file."""
    try:
        with open(settings_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Settings file {settings_file} not found. Using default settings.")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Settings file {settings_file} is not valid JSON. Using default settings.")
        return {}

def parse_args():
    # First load default settings
    settings_file = os.path.join(os.path.dirname(__file__), 'settings', 'defaults.json')
    settings = load_settings(settings_file)
    
    # Extract defaults from settings
    model_settings = settings.get('model', {})
    training_settings = settings.get('training', {})
    system_settings = settings.get('system', {})
    paths_settings = settings.get('paths', {})
    metrics_settings = settings.get('metrics', {})
    
    # Create argument parser with defaults from JSON
    parser = argparse.ArgumentParser(description='Zero-shot DSR Training')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=model_settings.get('latent_dim', 10),
                        help='Dimension of latent state')
    parser.add_argument('--hidden_dim', type=int, default=model_settings.get('hidden_dim', 50),
                        help='Dimension of hidden state for clipped shallow PLRNN')
    parser.add_argument('--experts', type=int, default=model_settings.get('experts', 10),
                        help='Number of experts')
    parser.add_argument('--pwl_units', type=int, default=model_settings.get('pwl_units', 2),
                        help='Number of PWL units')
    parser.add_argument('--expert_type', type=str, default=model_settings.get('expert_type', 'almost_linear_rnn'),
                        choices=['almost_linear_rnn', 'clipped_shallow_plrnn'],
                        help='Type of expert architecture to use')
    parser.add_argument('--probabilistic_expert', action='store_true', 
                        default=model_settings.get('probabilistic_expert', False),
                        help='Enable probabilistic experts with learnable noise')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=training_settings.get('batch_size', 16),
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=training_settings.get('epochs', 2000),
                        help='Number of epochs for training')
    parser.add_argument('--alpha', type=float, default=training_settings.get('alpha', 1.0),
                        help='Alpha parameter for teacher forcing')
    parser.add_argument('--n_interleave', type=int, default=training_settings.get('n_interleave', 10),
                        help='Interleave parameter for teacher forcing')
    parser.add_argument('--start_lr', type=float, default=training_settings.get('start_learning_rate', 2e-3),
                        help='Starting learning rate')
    parser.add_argument('--end_lr', type=float, default=training_settings.get('end_learning_rate', 1e-5),
                        help='Ending learning rate')
    parser.add_argument('--batches_per_epoch', type=int, default=training_settings.get('batches_per_epoch', 50),
                        help='Number of batches per epoch')
    parser.add_argument('--ssi', type=int, default=training_settings.get('scalar_saving_interval', 20),
                        help='Scalar saving interval')
    parser.add_argument('--noise_level', type=float, default=training_settings.get('noise_level', 0.05),
                        help='Noise level for training data')
    parser.add_argument('--reg_strength', type=float, default=training_settings.get('regularization_strength', 0.01),
                        help='Regularization strength for DynaMix model')
    
    # Metrics arguments
    ssd_settings = metrics_settings.get('state_space_divergence', {})
    ps_settings = metrics_settings.get('power_spectrum', {})
    mase_settings = metrics_settings.get('mase', {})
    
    parser.add_argument('--n_bins', type=int, default=ssd_settings.get('n_bins', 30),
                        help='Number of bins for state space divergence')
    parser.add_argument('--ps_smoothing', type=int, default=ps_settings.get('smoothing', 20),
                        help='Smoothing parameter for power spectrum')
    parser.add_argument('--mase_steps', type=int, default=mase_settings.get('steps', 20),
                        help='Number of steps for Mean Absolute Scaled Error')
    
    # System arguments
    parser.add_argument('--threads', type=int, default=system_settings.get('threads', 4),
                        help='Number of threads to use for computation')
    parser.add_argument('--seed', type=int, default=system_settings.get('seed', 42),
                        help='Random seed')
    parser.add_argument('--device', type=str, default=system_settings.get('device', 'cpu'), 
                        choices=['cpu', 'cuda'],
                        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--gpu_id', type=int, default=system_settings.get('gpu_id', 0), 
                help='ID of GPU to use (if multiple are available)')
    
    # Paths
    parser.add_argument('--data_path', type=str, default=paths_settings.get('data_path', ''),
                        help='Path to training data') # expected to be in form of a ($T-T_C+Δt+1$, $S$, $N$) NumPy array (.npy)
    parser.add_argument('--context_path', type=str, default=paths_settings.get('context_path', ''),
                        help='Path to context data') # expected to be in form of a ($T_C$, $S$, $N$) NumPy array (.npy)
    parser.add_argument('--test_path', type=str, default=paths_settings.get('test_path', ''),
                        help='Path to test data') # Optional: expected to be in form of a ($T$, $S$, $N$) NumPy array (.npy)
    parser.add_argument('--save_path', type=str, default=paths_settings.get('save_path', 'results'),
                        help='Path to save results')
    
    return parser.parse_args()

def training_setup():
    """Main function to run the Zero-shot DSR training."""
    # Parse command-line arguments
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set thread count for PyTorch
    torch.set_num_threads(args.threads)
    
    print(f"Zero-shot DSR training initialized with {args.threads} threads.")
    
    # Setup device
    device = setup_gpu(args)
    
    # Load all data
    print(f"Loading data from {args.data_path} (ground truth), {args.context_path} (context), {args.test_path} (test)")
    data = torch.tensor(np.load(args.data_path).astype(np.float32), device=device)
    context = torch.tensor(np.load(args.context_path).astype(np.float32), device=device)
    test = torch.tensor(np.load(args.test_path).astype(np.float32), device=device)
    dataset = Dataset(data, context, test, batch_size=args.batch_size, noise_level=args.noise_level, device=device)
    print(f"Data shape: {dataset.X.shape}, Context shape: {dataset.context.shape}, Test shape: {dataset.test.shape}")

    # Initialize DynaMix model
    if args.expert_type == "almost_linear_rnn":
        model = DynaMix(M=args.latent_dim, P=args.pwl_units, N=context.shape[2], Experts=args.experts, 
                        expert_type=args.expert_type, hidden_dim=args.hidden_dim, 
                        probabilistic_expert=args.probabilistic_expert).to(device)
    elif args.expert_type == "clipped_shallow_plrnn":
        model = DynaMix(M=args.latent_dim, hidden_dim=args.hidden_dim, N=context.shape[2], Experts=args.experts, 
                        expert_type=args.expert_type, probabilistic_expert=args.probabilistic_expert, 
                        ).to(device)
    else:
        raise ValueError(f"Unknown expert type: {args.expert_type}")
    print_model_parameters(model)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(args.end_lr/args.start_lr)/args.epochs))

    # Train DynaMix model
    train_dynamix(model, dataset, optimizer, scheduler, args)
    
    print("Training complete!")

if __name__ == "__main__":
    training_setup()