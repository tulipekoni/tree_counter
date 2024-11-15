import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.UNet import UNet
from torch.utils.data import DataLoader
from utils.arg_parser import parse_test_args
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

def create_scatter_plots(model, loader_A, loader_C, device):
    # Initialize lists for both regions
    ground_truth_A, predicted_A = [], []
    ground_truth_C, predicted_C = [], []
    
    # Collect predictions for region A
    with torch.no_grad():
        for batch_images, batch_labels, _ in loader_A:
            ground_truth_A.extend([len(p) for p in batch_labels])
            batch_images = batch_images.to(device)
            pred_maps = model(batch_images)
            predicted_A.extend(pred_maps.sum(dim=(1, 2, 3)).cpu().numpy())
    
    # Collect predictions for region C
    with torch.no_grad():
        for batch_images, batch_labels, _ in loader_C:
            ground_truth_C.extend([len(p) for p in batch_labels])
            batch_images = batch_images.to(device)
            pred_maps = model(batch_images)
            predicted_C.extend(pred_maps.sum(dim=(1, 2, 3)).cpu().numpy())
    
    # Convert to numpy arrays
    ground_truth_A, predicted_A = np.array(ground_truth_A), np.array(predicted_A)
    ground_truth_C, predicted_C = np.array(ground_truth_C), np.array(predicted_C)
    
    # Calculate metrics
    mae_A = np.mean(np.abs(predicted_A - ground_truth_A))
    rmse_A = np.sqrt(np.mean((predicted_A - ground_truth_A) ** 2))
    mae_C = np.mean(np.abs(predicted_C - ground_truth_C))
    rmse_C = np.sqrt(np.mean((predicted_C - ground_truth_C) ** 2))
    
    # Combine data for overall plot
    ground_truth_all = np.concatenate([ground_truth_A, ground_truth_C])
    predicted_all = np.concatenate([predicted_A, predicted_C])
    mae_all = np.mean(np.abs(predicted_all - ground_truth_all))
    rmse_all = np.sqrt(np.mean((predicted_all - ground_truth_all) ** 2))
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    
    # Helper function to plot each subplot
    def plot_scatter(ax, gt, pred, title, mae, rmse):
        # Original scatter plot
        ax.scatter(gt, pred, alpha=0.5, color='#404040')
        
        # Calculate and plot line of best fit
        z = np.polyfit(gt, pred, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(gt), max(gt), 100)
        ax.plot(x_range, p(x_range), 'b-', alpha=0.8, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
        
        # Plot ideal line (y=x)
        max_count = max(max(gt), max(pred))
        min_count = min(min(gt), min(pred))
        ax.plot([min_count, max_count], [min_count, max_count], 'r--', alpha=0.8, label='y = x')
        
        ax.set_xlabel('Ground Truth Count')
        ax.set_ylabel('Predicted Count')
        ax.set_title(f'{title}\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
    
    # Plot each region and combined data
    plot_scatter(ax1, ground_truth_A, predicted_A, 'Region A', mae_A, rmse_A)
    plot_scatter(ax2, ground_truth_C, predicted_C, 'Region C', mae_C, rmse_C)
    plot_scatter(ax3, ground_truth_all, predicted_all, 'Combined Regions', mae_all, rmse_all)
    
    plt.tight_layout()
    plt.savefig('scatter_plots_combined.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return mae_A, rmse_A, mae_C, rmse_C, mae_all, rmse_all

def main():
    args = parse_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.model_dir, device)
    
    # Define filter functions for regions
    def filter_A(filename): return filename.startswith("A_")
    def filter_C(filename): return filename.startswith("C_")
    
    # Create datasets and dataloaders for each region
    dataset_A = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), 
                                  filter_func=filter_A, augment=False)
    dataset_C = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), 
                                  filter_func=filter_C, augment=False)
    
    loader_A = DataLoader(dataset_A, batch_size=1, shuffle=False, 
                         num_workers=args.num_workers)
    loader_C = DataLoader(dataset_C, batch_size=1, shuffle=False, 
                         num_workers=args.num_workers)
    
    # Create scatter plots and get metrics
    mae_A, rmse_A, mae_C, rmse_C, mae_all, rmse_all = create_scatter_plots(
        model, loader_A, loader_C, device
    )
    
    # Print metrics for each region and combined
    print(f"Region A Metrics - MAE: {mae_A:.2f}, RMSE: {rmse_A:.2f}")
    print(f"Region C Metrics - MAE: {mae_C:.2f}, RMSE: {rmse_C:.2f}")
    print(f"Combined Metrics - MAE: {mae_all:.2f}, RMSE: {rmse_all:.2f}")

if __name__ == "__main__":
    main()
