import torch
import os
import matplotlib.pyplot as plt
import random
from datasets.tree_counting_dataset import TreeCountingDataset  
from models.unet import Unet
import argparse
from models.IndivBlur import IndivBlur
from utils.helper import GaussianKernel
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def find_checkpoint_file(folder_path):
    for file in os.listdir(folder_path):
        if file.startswith('checkpoint') and file.endswith('.tar'):
            return os.path.join(folder_path, file)
    raise FileNotFoundError(f"No checkpoint file found in {folder_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test and visualize the density map.')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the folder containing the model checkpoint and config')
    args = parser.parse_args()
    return args

def load_model_and_refiner(config, checkpoint_path, device):
    model = Unet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    kernel_size = checkpoint['kernel_size']

    # Load the model state from the saved checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move the model to the device & set to evaluation mode
    model.to(device)
    model.eval()

    # Check if 'refiner_state_dict' is in the checkpoint
    if 'refiner_state_dict' in checkpoint:
        refiner = IndivBlur(kernel_size=kernel_size, softmax=config['softmax'], downsample=config['downsample'])
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
        refiner.to(device)
        refiner.eval()
        use_refiner = True
        kernel_generator = None
    else:
        refiner = None
        use_refiner = False
        kernel_generator = GaussianKernel(kernel_size=kernel_size, downsample=config['downsample'], device=device, sigma=config['gaussian_sigma'])

    return model, refiner, kernel_size, use_refiner, kernel_generator

def visualize_input_and_density(img, model_prediction, ground_truth):
    # Convert the input image tensor to numpy for visualization (from C, H, W to H, W, C)
    img = img.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

    # Convert the density maps to numpy
    model_prediction = model_prediction.squeeze().cpu().detach().numpy()
    ground_truth = ground_truth.squeeze().cpu().detach().numpy()

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Plot the input image
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')  # Turn off axis

    # Plot the predicted density map
    im = axes[1].imshow(model_prediction, cmap='jet')
    axes[1].set_title("Predicted Density Map")
    axes[1].axis('off')  # Turn off axis
    fig.colorbar(im, ax=axes[1])

    # Plot the ground truth density map
    im2 = axes[2].imshow(ground_truth, cmap='jet')
    axes[2].set_title("Ground-truth Density Map")
    axes[2].axis('off')  # Turn off axis
    fig.colorbar(im2, ax=axes[2])

    # Show the plot
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    
    # Load config
    config_path = os.path.join(args.model_folder, 'config.json')
    config = load_config(config_path)

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(args.model_folder)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model & refiner from the checkpoint
    model, refiner, kernel_size, use_refiner, kernel_generator = load_model_and_refiner(config, checkpoint_path, device)
    
    # Load the test dataset
    test_dataset = TreeCountingDataset(root_path=os.path.join(config['data_dir'], 'test'))
    
    # Randomly select a test image from the dataset
    random_idx = random.randint(0, len(test_dataset) - 1)
    x, y, _ = test_dataset[random_idx]
    print(f"Selected random image index: {random_idx}")
    
    # Move the image to the device and add a batch dimension
    x = x.unsqueeze(0).to(device)
    y = [y.to(device)]  # Wrap y in a list to match the expected format
    
    # Run the model to get the density map
    with torch.no_grad():
        model_prediction = model(x)
        if use_refiner:
            ground_truth = refiner(y, x, model_prediction.shape)
        else:
            # Generate ground truth density map using Gaussian kernels
            ground_truth = kernel_generator.generate_density_map(y, model_prediction.shape)
    
    # Visualize and save both the input image and the density maps
    visualize_input_and_density(x.squeeze(0), model_prediction, ground_truth)