import argparse
import os
import torch
import json
from models.unet import Unet
from datasets.tree_counting_dataset import TreeCountingDataset
from torch.utils.data import DataLoader
import numpy as np
from utils.helper import GaussianKernel
from models.IndivBlur import IndivBlur
from utils.config_loader import load_config
from utils.checkpoint_utils import find_checkpoint_file, load_checkpoint
from utils.arg_parser import parse_test_args

def parse_args():
    parser = argparse.ArgumentParser(description='Test Tree Counting Model')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the folder containing the model checkpoint and config')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_test_args()

    # Load config
    config_path = os.path.join(args.model_folder, 'config.json')
    config = load_config(config_path)

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(args.model_folder)
    model_state_dict, refiner_state_dict, _, _, kernel_size, _ = load_checkpoint(checkpoint_path)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = Unet()
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # Initialize refiner or kernel generator
    if config['use_indivblur']:
        refiner = IndivBlur(kernel_size=kernel_size, softmax=config['softmax'], downsample=config['downsample'])
        refiner.load_state_dict(refiner_state_dict)
        refiner.to(device)
        refiner.eval()
    else:
        kernel_generator = GaussianKernel(kernel_size=kernel_size, downsample=config['downsample'], device=device)

    # Load test dataset
    test_dataset = TreeCountingDataset(root_path=os.path.join(config['data_dir'], 'test'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config['num_workers'])

    # Testing loop
    mae_list = []
    rmse_list = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = [p.to(device) for p in y]

            # Forward pass
            outputs = model(x)

            # Generate ground truth density maps
            if config['use_indivblur']:
                pred = refiner(y, x, outputs.shape)
            else:
                pred = kernel_generator.generate_density_map(y, outputs.shape)

            # Calculate metrics
            gt_count = pred.sum().item()
            pred_count = outputs.sum().item()
            mae = abs(gt_count - pred_count)
            rmse = (gt_count - pred_count) ** 2

            mae_list.append(mae)
            rmse_list.append(rmse)

    # Calculate final metrics
    final_mae = np.mean(mae_list)
    final_rmse = np.sqrt(np.mean(rmse_list))

    print(f"Test Results: MAE = {final_mae:.2f}, RMSE = {final_rmse:.2f}")

