import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.model_loader import load_weights
from utils.arg_parser import parse_test_args
from utils.helper import RunningAverageTracker
from datasets.tree_counting_dataset import TreeCountingDataset

def test_model(model, dataloader, device):
    test_mae = RunningAverageTracker()
    test_rmse = RunningAverageTracker()

    with torch.no_grad():
        for step, (batch_images, batch_labels, batch_names) in enumerate(dataloader):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=device)
            batch_images = batch_images.to(device)
            batch_pred_density_maps = model(batch_images)

            # The number of trees is total sum of all prediction pixels
            batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()
            batch_differences = batch_pred_counts - batch_gt_count

            # Update MAE and RMSE metrics
            batch_size = batch_pred_counts.shape[0]
            test_mae.update(torch.abs(batch_differences).sum().item(), n=batch_size)
            test_rmse.update(torch.sum(batch_differences ** 2).item(), n=batch_size)

    average_mae = test_mae.get_average()
    average_rmse = torch.sqrt(torch.tensor(test_rmse.get_average())).item()
       
    return average_mae, average_rmse

def main():
    args = parse_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, _ = load_weights(args.model_dir, device)

    def filter_A(filename):
        return filename.startswith("A_")
    
    def filter_C(filename):
        return filename.startswith("C_")  

    # Prepare datasets and dataloaders
    dataset_A = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_A, augment=False)
    dataset_C = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_C, augment=False)

    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False, num_workers=args.num_workers)
    loader_C = DataLoader(dataset_C, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Test on region A
    mae_A, rmse_A = test_model(model, loader_A, device)
    print(f"Region A - MAE: {mae_A:.2f}, RMSE: {rmse_A:.2f}")

    # Test on region C
    mae_C, rmse_C = test_model(model, loader_C, device)
    print(f"Region C - MAE: {mae_C:.2f}, RMSE: {rmse_C:.2f}")

    # Calculate combined metrics
    total_count = len(dataset_A) + len(dataset_C)
    mae_combined = (mae_A * len(dataset_A) + mae_C * len(dataset_C)) / total_count
    rmse_combined = np.sqrt((rmse_A**2 * len(dataset_A) + rmse_C**2 * len(dataset_C)) / total_count)
    print(f"All - MAE: {mae_combined:.2f}, RMSE: {rmse_combined:.2f}")

if __name__ == "__main__":
    main()