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

def combined_loss(output, target):
    """
    Calculate combined loss using MAE and cosine similarity.
    
    Args:
        output (torch.Tensor): Model output tensor
        target (torch.Tensor): Ground truth tensor
        
    Returns:
        torch.Tensor: Combined loss value
    """
    output_count = output.sum(dim=(1, 2, 3))
    target_count = target.sum(dim=(1, 2, 3))

    mse_loss = F.mse_loss(output_count, target_count)
    cos_loss_val = cos_loss(output, target) 
    alpha = 10
    total_loss = mse_loss + alpha * cos_loss_val

    return total_loss
