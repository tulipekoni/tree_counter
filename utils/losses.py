import torch
from torch.nn import MSELoss
import torch.nn.functional as F

def cos_loss(output, target):
    """
    Calculate cosine similarity loss between output and target tensors.
    
    Args:
        output (torch.Tensor): Model output tensor
        target (torch.Tensor): Ground truth tensor
        
    Returns:
        torch.Tensor: Cosine similarity loss
    """
    B = output.shape[0]
    output = output.reshape(B, -1)
    target = target.reshape(B, -1)
    loss = torch.mean(1 - F.cosine_similarity(output, target))
    return loss


def combined_loss(output, target, sigma=None):
    """
    Calculate loss for density maps where sum represents object count.
    Returns both total loss and individual components for logging.
    """
    pixel_multiplier = 0.3
    cos_multiplier = 2
    
    batch_size = output.shape[0]
    
    pixel_loss = pixel_multiplier * (torch.abs(output - target).sum() / batch_size)

    # Count prediction error
    pred_counts = output.sum(dim=(1,2,3))
    true_counts = target.sum(dim=(1,2,3))
    count_loss = torch.abs(pred_counts - true_counts).mean()

    # Cosine similarity for structural similarity
    cos_loss_val = cos_multiplier * cos_loss(output, target)
    
    # Modify sigma penalty handling
    sigma_penalty = 0
    if sigma is not None:
        sigma_weight = 0.01
        sigma_penalty = sigma_weight * sigma.pow(2).mean() 
    
    # Calculate total loss
    total_loss = pixel_loss + count_loss + cos_loss_val + sigma_penalty
    
    # Create base loss components dictionary
    loss_components = {
        'pixel_loss': pixel_loss.item(),
        'count_loss': count_loss.item(),
        'cos_loss': cos_loss_val.item(),
    }
    
    # Only add sigma_penalty to dictionary if sigma was provided
    if sigma is not None:
        loss_components['sigma_penalty'] = sigma_penalty.item()
    
    return total_loss, loss_components