import torch
import os
import numpy as np
from datasets.tree_counting_dataset import TreeCountingDataset  
from models.unet import Unet
import argparse
from models.IndivBlur import IndivBlur 
from utils.helper import GaussianKernel

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default='./processed_data', help='Directory containing the test data.')
    parser.add_argument('--save-dir', default='./checkpoints/model.pth', help='Path to the saved model.')
    parser.add_argument('--device', default='0', help='GPU device to use (e.g., "0").')
    args = parser.parse_args()
    return args

def load_model_and_refiner(args, device):
    model = Unet()
    checkpoint = torch.load(args.save_dir, map_location=device)
    kernel_size = checkpoint['kernel_size']

    # Load the model state from the saved checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move the model to the device & set to evaluation mode
    model.to(device)
    model.eval()

    # Check if 'refiner_state_dict' is in the checkpoint
    if 'refiner_state_dict' in checkpoint:
        refiner = IndivBlur(kernel_size=kernel_size, softmax=args.softmax, downsample=1)
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
        refiner.to(device)
        refiner.eval()
        use_refiner = True
        kernel_generator = None
    else:
        refiner = None
        use_refiner = False
        kernel_generator = GaussianKernel(kernel_size=kernel_size, downsample=1, device=device)

    print(use_refiner)
    return model, refiner, kernel_size, use_refiner, kernel_generator

if __name__ == '__main__':
    args = parse_args()

    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the dataset and dataloader
    test_dataset = TreeCountingDataset(root_path=os.path.join(args.data_dir))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Load the model
    model, refiner, kernel_size, use_refiner, kernel_generator = load_model_and_refiner(args, device)

    # Store the differences between predictions and ground truth
    epoch_minus = []
    
    # Perform inference on the test dataset
    with torch.no_grad():
        for x, y in test_dataloader:
            x = x.to(device)
            y = [p.to(device) for p in y]

            assert input.size(0) == 1, 'Batch size should be 1 for testing'

            # Forward pass through the model
            model_prediction = model(input)

            print('use_refiner', use_refiner)
            if use_refiner:
                ground_truth = refiner(y, x, model_prediction.shape)
            else:
                ground_truth = kernel_generator.generate_density_map(y, model_prediction.shape)
            
            # Calculate the difference between the predicted and ground truth counts
            pred_count = torch.sum(model_prediction).item()
            true_count = len(ground_truth[0])
            temp_minu = true_count - pred_count

            print(f"Predicted = {pred_count:.2f}, Ground Truth = {true_count}, Difference = {temp_minu:.2f}")
            epoch_minus.append(temp_minu)

    # Convert to numpy array
    epoch_minus = np.array(epoch_minus)

    # Calculate the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    mae = np.mean(np.abs(epoch_minus))
    rmse = np.sqrt(np.mean(np.square(epoch_minus)))

    # Print final results
    print(f"Final Test Results: MAE = {mae:.2f}, RMSE = {rmse:.2f}")
