import os
import torch
from typing import Tuple, Optional, Dict, Any

def find_checkpoint_file(folder_path: str) -> str:
    """
    Find the checkpoint file in the given folder.

    Args:
        folder_path (str): Path to the folder containing the checkpoint.

    Returns:
        str: Path to the found checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint file is found in the folder.
    """
    for file in os.listdir(folder_path):
        if file.startswith('checkpoint') and file.endswith('.tar'):
            return os.path.join(folder_path, file)
    raise FileNotFoundError(f"No checkpoint file found in {folder_path}")

def load_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], int, int]:
    """
    Load a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        Tuple containing:
        - model_state_dict (Dict[str, Any]): State dict of the model.
        - refiner_state_dict (Optional[Dict[str, Any]]): State dict of the refiner (if present).
        - optimizer_state_dict (Dict[str, Any]): State dict of the optimizer.
        - refiner_optimizer_state_dict (Optional[Dict[str, Any]]): State dict of the refiner optimizer (if present).
        - lr_scheduler_state_dict (Optional[Dict[str, Any]]): State dict of the learning rate scheduler (if present).
        - refiner_lr_scheduler_state_dict (Optional[Dict[str, Any]]): State dict of the refiner learning rate scheduler (if present).
        - kernel_size (int): Kernel size used in the checkpoint.
        - epoch (int): Epoch number of the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file is not found.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_state_dict = checkpoint['model_state_dict']
    refiner_state_dict = checkpoint.get('refiner_state_dict')
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    refiner_optimizer_state_dict = checkpoint.get('refiner_optimizer_state_dict')
    lr_scheduler_state_dict = checkpoint.get('lr_scheduler_state_dict')
    refiner_lr_scheduler_state_dict = checkpoint.get('refiner_lr_scheduler_state_dict')
    kernel_size = checkpoint['kernel_size']
    epoch = checkpoint['epoch']

    return model_state_dict, refiner_state_dict, optimizer_state_dict, refiner_optimizer_state_dict, lr_scheduler_state_dict, refiner_lr_scheduler_state_dict, kernel_size, epoch

def save_checkpoint(save_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    refiner: Optional[torch.nn.Module] = None, refiner_optimizer: Optional[torch.optim.Optimizer] = None, 
                    refiner_lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    kernel_size: int = 0, epoch: int = 0) -> str:
    """
    Save a checkpoint.

    Args:
        save_dir (str): Directory to save the checkpoint.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The learning rate scheduler to save.
        refiner (Optional[torch.nn.Module]): The refiner model to save (if present).
        refiner_optimizer (Optional[torch.optim.Optimizer]): The refiner optimizer to save (if present).
        refiner_lr_scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The refiner learning rate scheduler to save.
        kernel_size (int): Kernel size to save.
        epoch (int): Current epoch number.

    Returns:
        str: Path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.tar')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'kernel_size': kernel_size,
        'epoch': epoch
    }
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    if refiner is not None:
        checkpoint['refiner_state_dict'] = refiner.state_dict()
    if refiner_optimizer is not None:
        checkpoint['refiner_optimizer_state_dict'] = refiner_optimizer.state_dict()
    if refiner_lr_scheduler is not None:
        checkpoint['refiner_lr_scheduler_state_dict'] = refiner_lr_scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path