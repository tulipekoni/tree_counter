import os
import torch
from models.UNet import UNet
from models.DMG import DMG

def load_weights(checkpoint_path, device):
    """
    Load model weights from a checkpoint directory.
    
    Args:
        checkpoint_path (str): Path to directory containing checkpoint files
        device (torch.device): Device to load the model on
    
    Returns:
        model (UNet): Loaded model
        sigma (float): Sigma value from checkpoint
    """
    # Find the most recent .tar or .pth file
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith(('.tar', '.pth'))]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_path, f)))
    checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize model
    model = UNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get sigma value (default to 15 if not present)
    sigma = checkpoint.get('sigma', 15)
    
    return model, sigma

def load_weights_with_dmg(checkpoint_path, device):
    """
    Load model weights and create DMG model from a checkpoint directory.
    
    Args:
        checkpoint_path (str): Path to directory containing checkpoint files
        device (torch.device): Device to load the model on
    
    Returns:
        model (UNet): Loaded model
        dmg (DMG): Density map generator
    """
    model, sigma = load_weights(checkpoint_path, device)
    dmg = DMG(device=device, initial_sigma_value=sigma)
    dmg.to(device)
    dmg.eval()
    
    return model, dmg
