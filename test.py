import os
import torch
import numpy as np
from models.UNet import UNet
from torch.utils.data import DataLoader
from utils.arg_parser import parse_test_args
from models.StaticRefiner import StaticRefiner
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
    refiner = StaticRefiner(device=device, sigma=checkpoint['sigma'])
    model.to(device)
    refiner.to(device)
    model.eval()
    refiner.eval()
    return model, refiner

def test_model(model, dataloader, refiner, device):
    mae_sum = 0
    rmse_sum = 0
    count = 0

    with torch.no_grad():
        for images, labels, _ in enumerate(dataloader, desc="Testing"):
            images = images.to(device)
            labels = [label.to(device) for label in labels]
            gt_count = torch.tensor([len(p) for p in labels], dtype=torch.float32, device=device)

            pred_density_maps = model(images)
            pred_counts = pred_density_maps.sum(dim=(1, 2, 3))

            differences = pred_counts - gt_count
            mae_sum += torch.abs(differences).sum().item()
            rmse_sum += torch.sum(differences ** 2).item()
            count += len(gt_count)

    mae = mae_sum / count
    rmse = np.sqrt(rmse_sum / count)
    return mae, rmse

def main():
    args = parse_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, refiner = load_model(args.model_dir, device)

    def filter_A(filename):
        return filename.startswith("A_")
    
    def filter_B(filename):
        return filename.startswith("B")  

    # Prepare datasets and dataloaders
    dataset_A = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_A)
    dataset_C = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), filter_func=filter_B)

    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False, num_workers=args.num_workers)
    loader_C = DataLoader(dataset_C, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Test on region A
    mae_A, rmse_A = test_model(model, loader_A, refiner, device)

    # Test on region C
    mae_C, rmse_C = test_model(model, loader_C, refiner, device)

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