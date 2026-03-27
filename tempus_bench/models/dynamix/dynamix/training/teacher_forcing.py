import torch

def predict_sequence_using_gtf(model, x, context, alpha, n_interleave):
    """
    Predicts a sequence using teacher forcing (for training)
    
    Args:
        model: DynaMix model
        x: Input sequence tensor of shape (seq_length, batch_size, N)
        context: Context tensor of shape (seq_length, batch_size, N)
        alpha: Teacher forcing strength (0-1)
        n_interleave: Apply teacher forcing every n_interleave steps
    
    Returns:
        Predicted sequence of shape (seq_length, batch_size, M)
    """
    seq_length, batch_size, N = x.shape
    device = x.device
    
    # Initialize prediction tensor on the same device
    Z = torch.empty(seq_length, batch_size, model.M, device=device)
    
    # Initialize latent state from first observation
    first_x = x[0]  # (batch_size, N)
    z = torch.matmul(first_x, model.B).t()  # (M, batch_size)
    z[:N, :] = first_x.t()  # Copy observation directly to first N latent dimensions
    
    # Apply teacher forcing to the initial state
    z = teacher_force(z, first_x, alpha=1.0, N=N)
    
    # Pre-calculate which timesteps need teacher forcing
    tf_steps = torch.tensor([t % n_interleave == 0 and t > 0 for t in range(seq_length)], device=device)
    
    # Generate sequence predictions
    for t in range(seq_length):
        # Apply teacher forcing at regular intervals
        if tf_steps[t]:
            z = teacher_force(z, x[t], alpha, N=N)
            
        # Update the latent state using the model
        z = model(z, context)
        Z[t] = z.t()  # Store transposed z: (batch_size, M)
    
    return Z

def teacher_force(z, x, alpha, N):
    """
    Apply teacher forcing to the latent state
    
    Args:
        z: Latent state tensor of shape (M, batch_size)
        x: Ground truth observation of shape (batch_size, N)
        alpha: Teacher forcing strength (0-1)
        N: Observation dimension
    
    Returns:
        Teacher-forced latent state of shape (M, batch_size)
    """
    # Teacher force the first N dimensions of the state
    # z: (M, batch_size), x: (batch_size, N)
    z_obs = z[:N, :]  # (N, batch_size)
    x_t = x.t()  # (N, batch_size)
    
    # Use in-place operation when alpha=1.0 for efficiency
    if alpha == 1.0:
        z[:N, :] = x_t
    else:
        z[:N, :] = alpha * x_t + (1 - alpha) * z_obs
        
    return z