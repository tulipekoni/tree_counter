import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
from models.UNet import UNet
from torch.utils.data import DataLoader
from utils.arg_parser import parse_test_args
from utils.helper import RunningAverageTracker
from datasets.tree_counting_dataset import TreeCountingDataset
import gc
import math


def load_model(checkpoint_path, device):
    model = UNet()
    
    # Find the most recent .tar file in the resume folder
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.tar')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No .tar checkpoint files found in {checkpoint_path}")
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_path, f)))
    checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def test_model(model, dataloader, device):
    test_mae = RunningAverageTracker()
    test_rmse = RunningAverageTracker()
    
    # Move model to eval mode
    model.eval()
    
    with torch.no_grad(), torch.cuda.amp.autocast():  # Enable automatic mixed precision
        for step, (batch_images, batch_labels, batch_names) in enumerate(dataloader):
            try:
                # Process one image at a time
                for i in range(len(batch_images)):
                    # Process single image
                    single_image = batch_images[i:i+1].to(device)
                    gt_count = torch.tensor([len(batch_labels[i])], 
                                          dtype=torch.float32, 
                                          device='cpu')
                    
                    # Forward pass
                    pred_density_map = model(single_image)
                    pred_count = pred_density_map.sum(dim=(1,2,3)).cpu()
                    
                    # Calculate metrics on CPU
                    difference = pred_count - gt_count
                    
                    # Update metrics
                    test_mae.update(torch.abs(difference).sum().item(), n=1)
                    test_rmse.update(torch.sum(difference ** 2).item(), n=1)
                    
                    # Clear memory
                    del single_image, pred_density_map, pred_count
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"Error processing batch at step {step}: {e}")
                continue
            
            if step % 10 == 0:  # Print progress every 10 batches
                print(f"Processed {step} batches")
                
            # Aggressive memory clearing
            gc.collect()
            torch.cuda.empty_cache()

    mae = test_mae.get_average()
    rmse = math.sqrt(test_rmse.get_average())
    
    return mae, rmse

def main():
    args = parse_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model_dir, device)

    def filter_A(filename):
        return filename.startswith("A_")
    
    def filter_C(filename):
        return filename.startswith("C_")  

    # Prepare datasets and dataloaders
    dataset_A = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_A, augment=False)
    dataset_C = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_C, augment=False)

    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False, num_workers=args.num_workers)
    loader_C = DataLoader(dataset_C, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Try to optimize CUDA memory allocation
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available GPU memory
    torch.backends.cudnn.benchmark = True
    
    # Clear memory before starting
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Test on region A
        mae_A, rmse_A = test_model(model, loader_A, device)
        
        # Aggressive memory clearing between regions
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reload model for region C
        model = load_model(args.model_dir, device)
        model = model.to(device)
        
        # Test on region C
        mae_C, rmse_C = test_model(model, loader_C, device)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    
    # Calculate combined metrics
    total_count = len(dataset_A) + len(dataset_C)
    mae_combined = (mae_A * len(dataset_A) + mae_C * len(dataset_C)) / total_count
    rmse_combined = np.sqrt((rmse_A**2 * len(dataset_A) + rmse_C**2 * len(dataset_C)) / total_count)
    print(f"All - MAE: {mae_combined:.2f}, RMSE: {rmse_combined:.2f}")

if __name__ == "__main__":
    main()