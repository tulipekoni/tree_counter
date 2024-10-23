import os
import torch
import numpy as np
from models.UNet import UNet
from torch.utils.data import DataLoader
from utils.arg_parser import parse_test_args
from models.StaticRefiner import StaticRefiner
from utils.helper import RunningAverageTracker
from datasets.tree_counting_dataset import TreeCountingDataset


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
    epoch_mae = RunningAverageTracker()
    epoch_rmse = RunningAverageTracker()

    with torch.no_grad():
        for step, (batch_images, batch_labels, batch_names) in enumerate(dataloader):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=device)
            batch_images = batch_images.to(device)
            batch_pred_density_maps = model(batch_images)

            # The number of trees is total sum of all prediction pixels
            batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()
            batch_differences = batch_pred_counts - batch_gt_count

            # Update loss, MAE, and RMSE metrics
            batch_size = batch_pred_counts.shape[0]
            epoch_mae.update(torch.mean(torch.abs(batch_differences)).item(), batch_size)
            epoch_rmse.update(torch.sqrt(torch.mean(batch_differences ** 2)).item(), batch_size)

    average_rmse = epoch_rmse.get_average()
    average_mae = epoch_mae.get_average()
       
    return average_mae, average_rmse

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
    dataset_A = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_A)
    dataset_C = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_C)

    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False, num_workers=args.num_workers)
    loader_C = DataLoader(dataset_C, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Test on region A
    mae_A, rmse_A = test_model(model, loader_A, device)

    # Test on region C
    mae_C, rmse_C = test_model(model, loader_C, device)

    # Calculate combined metrics
    total_count = len(dataset_A) + len(dataset_C)
    mae_combined = (mae_A * len(dataset_A) + mae_C * len(dataset_C)) / total_count
    rmse_combined = np.sqrt((rmse_A**2 * len(dataset_A) + rmse_C**2 * len(dataset_C)) / total_count)


    # Print results
    print(f"Region A - MAE: {mae_A:.2f}, RMSE: {rmse_A:.2f}")
    print(f"Region C - MAE: {mae_C:.2f}, RMSE: {rmse_C:.2f}")
    print(f"Combined - MAE: {mae_combined:.2f}, RMSE: {rmse_combined:.2f}")
    print(f"Combined MAE + RMSE: {mae_combined + rmse_combined:.2f}")

if __name__ == "__main__":
    main()