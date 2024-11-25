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

def modified_softmax(x, t=20.0, b=3.0):
    """
    Modified softmax function that applies softplus with an offset.
    
    Args:
    - x (torch.Tensor): Input tensor
    - t (float): Threshold/offset value
    - b (float): Base offset value added to the result
    
    Returns:
    - torch.Tensor: The output tensor after applying softplus with offset
    """
    return b + torch.log(1 + torch.exp(x - t))

def combined_loss(output, target, coss_loss_multiplier=1.0):
    """
    Calculate loss for density maps where sum represents object count.
    Returns both total loss and individual components for logging.
    """
    batch_size = output.shape[0]
    
    # MSE focused on non-zero regions
    mse_criterion = MSELoss(reduction='sum')
    pixel_loss = mse_criterion(output, target) / batch_size
    
    # Count prediction error
    pred_counts = output.sum(dim=(1,2,3))
    true_counts = target.sum(dim=(1,2,3))
    count_loss = torch.abs(pred_counts - true_counts).mean()
    
    # Cosine similarity for structural similarity
    cos_loss_val = cos_loss(output, target)
    
    # Calculate total loss
    total_loss = pixel_loss + count_loss + coss_loss_multiplier * cos_loss_val
    
    return total_loss, {
        'pixel_loss': pixel_loss.item(),
        'count_loss': count_loss.item(),
        'cos_loss': (coss_loss_multiplier * cos_loss_val).item()
    }
