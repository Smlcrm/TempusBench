import torch
import copy
import os
import json
import time
from pathlib import Path
from ..metrics.metrics import geometrical_misalignment, temporal_misalignment, MASE
from ..model.forecaster import DynaMixForecaster
from .teacher_forcing import predict_sequence_using_gtf
from ..utilities.utilities import save_model, create_checkpoint_directories
from ..utilities.plotting import plot_trajectories, plot_metrics

def loss_function(model, predictions, targets, lambda_reg=0.1, c=0.01):
    """
    Compute the loss.
    
    Args:
        model: DynaMix model
        predictions: Predicted values from the model
        targets: Target values
        lambda_reg: Regularization strength
        
    Returns:
        Combined loss with regularization
    """
    # Extract observation dimensions
    n_dims = model.N
    
    # Compute MSE loss
    mse_loss = torch.nn.functional.mse_loss(predictions[:,:,:n_dims], targets)
    
    # Initialize regularization loss
    reg_loss = 0.0
    
    # Apply latent model regularization
    if lambda_reg > 0:
        sigma_values = model.gating_network.sigma
        reg_loss += lambda_reg * torch.mean(torch.exp(-torch.abs(sigma_values)/c))
        
    # Combine losses
    total_loss = mse_loss + reg_loss
    
    return total_loss

def train_dynamix(model, dataset, optimizer, scheduler, args, printing=True, plotting=True):
    """
    Train using sparse teacher forcing.
    
    Args:
        model: DynaMix model
        dataset: Dataset object
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        args: Namespace containing all arguments
        printing: Whether to print training progress
        plotting: Whether to generate trajectory plots at SSI intervals
    """
    # If args is provided, use those parameters
    num_epochs = args.epochs
    alpha = args.alpha
    n_interleave = args.n_interleave
    batches_per_epoch = args.batches_per_epoch
    ssi = args.ssi
    n_bins = args.n_bins
    ps_smoothing = args.ps_smoothing
    mase_steps = args.mase_steps
    lambda_reg = args.reg_strength

    # Testing hyperparameters
    context_length = 2000
    prediction_steps = 10000
    plot_id = 0
    
    # Setup checkpoint directories using the utility function
    save_dir, checkpoint_dir, plots_dir = create_checkpoint_directories(args.save_path, args)
    
    print(f"Starting training for {num_epochs} epochs.")
    
    model.train()  # Set model to training mode
    losses = []
    klx = []
    dh = []
    mase_values = []
    ssi_epochs = []
    epoch_times = []  # Track epoch times
    
    epochs = range(num_epochs)

    for e in epochs:
        epoch_start_time = time.time()
        epoch_losses = []
        
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()  # Reset gradients for the optimizer
            
            x, y, context = dataset.sample_batch()  # Sample a batch of data (x: inputs, y: targets)
            z_hat = predict_sequence_using_gtf(model, x, context, alpha, n_interleave)  # Predict sequence using teacher forcing
            
            # Calculate loss
            loss = loss_function(
                model, 
                z_hat, 
                y, 
                lambda_reg=lambda_reg
            )
            
            loss.backward()  # Backward pass
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            epoch_losses.append(loss.item())
    
        scheduler.step()  # Adjust learning rate based on the scheduler
        
        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        
        # Compute and store average loss for the epoch
        average_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(average_epoch_loss)

        # Calculate and print metrics at SSI intervals, save model checkpoints
        if e % ssi == 0:
            with torch.no_grad():
                # Temporarily switch model to eval mode for prediction
                model.eval()
                
                # Generate predictions
                forecaster = DynaMixForecaster(model)
                X_gen = forecaster.forecast(dataset.test[0:context_length,:,:], prediction_steps)
                
                # Move tensors to CPU for metrics calculation
                X_gen_cpu = X_gen.cpu() if X_gen.is_cuda else X_gen
                test_data_cpu = dataset.test.clone().detach().cpu() if dataset.test.is_cuda else dataset.test.clone().detach()
                
                klx_values = []
                dh_values = []
                mase_batch_values = []
                # Calculate metrics using parameters from settings
                for eval_id in range(dataset.test.shape[1]):
                    klx_values.append(geometrical_misalignment(
                        X_gen_cpu[:,eval_id,:], test_data_cpu[context_length:context_length+prediction_steps,eval_id,:], 
                        n_bins=n_bins
                    ))
                    dh_values.append(temporal_misalignment(
                        X_gen_cpu[:,eval_id,:], test_data_cpu[context_length:context_length+prediction_steps,eval_id,:], 
                        smoothing=ps_smoothing
                    ))
                    mase_batch_values.append(MASE(
                        test_data_cpu[context_length:context_length+prediction_steps,eval_id,:], X_gen_cpu[:,eval_id,:], 
                        steps=mase_steps
                    ))
                
                # Store metrics
                klx.append(torch.nanmedian(torch.tensor(klx_values, device=model.B.device)))
                dh.append(torch.nanmedian(torch.tensor(dh_values, device=model.B.device)))
                mase_values.append(torch.nanmedian(torch.tensor(mase_batch_values, device=model.B.device)))
                ssi_epochs.append(e)
                epoch_times.append(epoch_duration)  # Store epoch time
                
                # Save model checkpoint
                save_model(checkpoint_dir / f"checkpoint_epoch_{e}.pt", model)
                
                # Generate simple trajectory plot if plotting is enabled
                if plotting:
                    # Move data to CPU for plotting
                    real_trajectory = dataset.test[context_length:context_length+prediction_steps, plot_id, :].cpu()
                    predicted_trajectory = X_gen[:,plot_id,:].cpu()
                    plot_trajectories(real_trajectory, predicted_trajectory, plots_dir, f"epoch_{e}_")
                
                # Switch back to train mode
                model.train()
            
            if printing:
                # Format epoch duration as minutes:seconds
                minutes = int(epoch_duration // 60)
                seconds = int(epoch_duration % 60)
                time_str = f"{minutes}m {seconds}s"
                
                # Calculate estimated remaining time
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = num_epochs - e - 1
                est_remaining_time = avg_epoch_time * remaining_epochs
                
                eta_hours = int(est_remaining_time // 3600)
                eta_mins = int((est_remaining_time % 3600) // 60)
                eta_secs = int(est_remaining_time % 60)
                eta_str = f"{eta_hours}h {eta_mins}m {eta_secs}s"
                
                print(f"Epoch {e} loss: {average_epoch_loss:.6f} | Epoch time: {time_str} | ETA: {eta_str} | Dstsp: {klx[-1]:.6f} | DH: {dh[-1]:.6f} | MASE: {mase_values[-1]:.6f}")

    # At the end of training, save final model
    final_checkpoint = checkpoint_dir / "final_model.pt"
    save_model(final_checkpoint, model)
    
    # Save complete metrics history at the end of training
    metrics_data = {
        'epochs': ssi_epochs,
        'losses': losses[::ssi][:len(ssi_epochs)],
        'dstsp': klx,
        'dh': dh,
        'pe': mase_values,
        'epoch_times': epoch_times  # Save epoch times
    }
    torch.save(metrics_data, checkpoint_dir / "metrics.pt")
    
    # Generate plots only at the end of training
    model.eval()
    with torch.no_grad():
        # Generate final predictions
        forecaster = DynaMixForecaster(model)
        X_gen = forecaster.forecast(dataset.test[0:context_length,:,:], prediction_steps)
        
        # Move tensors to CPU for plotting     
        X_gen_cpu = X_gen.cpu() if X_gen.is_cuda else X_gen
        real_trajectory = dataset.test[context_length:context_length+prediction_steps, plot_id, :].cpu()
        predicted_trajectory = X_gen_cpu[:,plot_id,:]
        
        # Plot metrics
        plot_metrics(metrics_data, plots_dir, "final_")
        
        # Plot trajectories
        plot_trajectories(real_trajectory, predicted_trajectory, plots_dir, "final_")