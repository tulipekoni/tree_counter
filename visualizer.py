import os
import cv2
import torch
import random
import numpy as np
from models.DMG import DMG
from models.UNet import UNet
from utils.losses import combined_loss
from torch.utils.data import DataLoader
from utils.arg_parser import parse_visualizer_args
from datasets.tree_counting_dataset import TreeCountingDataset
from utils.model_loader import load_weights_with_dmg

def main():
    # Parse command-line arguments
    args = parse_visualizer_args()

    # Setup device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and dmg
    model, dmg = load_weights_with_dmg(args.model_dir, device)

    # Prepare dataset and dataloader
    dataset = TreeCountingDataset(root_path=os.path.join(args.data_dir, 'test'), augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    while True:
        # Get a batch from the dataloader
        for batch_images, batch_labels, batch_names in dataloader:
            # Move data to device
            batch_images = batch_images.to(device)
            batch_labels = [labels.to(device) for labels in batch_labels]
            name = batch_names[0]  # Get the first (and only) filename since batch_size=1

            # Generate model prediction
            with torch.no_grad():
                pred_density_map = model(batch_images)

            # Generate ground truth density map using dmg
            gt_density_map = dmg(batch_images, batch_labels)

            # Calculate predicted count and ground truth count
            pred_count = pred_density_map.sum().item()
            gt_count = gt_density_map.sum().item()
            difference = abs(pred_count - gt_count)

            # Prepare images for display
            # Convert tensors to numpy arrays
            image_np = batch_images.cpu().squeeze().permute(1, 2, 0).numpy()
            pred_density_map_np = pred_density_map.cpu().squeeze().numpy()
            gt_density_map_np = gt_density_map.cpu().detach().squeeze().numpy()

            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])  # ImageNet mean
            std = np.array([0.229, 0.224, 0.225])   # ImageNet std
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Normalize density maps for display
            pred_density_map_display = pred_density_map_np / (pred_density_map_np.max() + 1e-9)
            pred_density_map_display = (pred_density_map_display * 255).astype(np.uint8)
            pred_density_map_display = cv2.applyColorMap(pred_density_map_display, cv2.COLORMAP_JET)

            gt_density_map_display = gt_density_map_np / (gt_density_map_np.max() + 1e-9)
            gt_density_map_display = (gt_density_map_display * 255).astype(np.uint8)
            gt_density_map_display = cv2.applyColorMap(gt_density_map_display, cv2.COLORMAP_JET)

            # Add labels and counts to the images
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Smaller text
            font_thickness = 2
            padding = 30  # Slightly reduced padding
            
            # Create blank space for text
            text_space = np.ones((padding, image_np.shape[1], 3), dtype=np.uint8) * 255
            image_with_label = np.vstack([text_space, image_np])
            gt_with_label = np.vstack([text_space, gt_density_map_display])
            pred_with_label = np.vstack([text_space, pred_density_map_display])
            
            # Function to center text
            def put_centered_text(img, text, y_pos):
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                cv2.putText(img, text, (text_x, y_pos), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
            # Add centered text to each image
            put_centered_text(image_with_label, 'subimage', 20)
            put_centered_text(gt_with_label, f'Ground Truth (Count: {gt_count:.1f})', 20)
            put_centered_text(pred_with_label, f'Prediction (Count: {pred_count:.1f})', 20)
            
            # Combine images side by side
            combined_image = np.hstack((image_with_label, gt_with_label, pred_with_label))

            # Calculate losses
            loss = combined_loss(pred_density_map, gt_density_map)

            # Display images and metrics
            cv2.imshow('Tree Counting Visualizer', combined_image)
            print(f'Image: {name}')
            print(f'Ground Truth Count: {gt_count:.2f}')
            print(f'Predicted Count: {pred_count:.2f}')
            print(f'Difference: {difference:.2f}')
            print("Press Enter to display a new image, press S to save, or press Esc to exit.")

            key = cv2.waitKey(0)
            if key == 27:  # Esc key to stop
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):  # S key to save
                save_path = f'visualization_{name}'
                cv2.imwrite(save_path, combined_image)
                print(f'Saved visualization to {save_path}')
            elif key == 13:  # Enter key to continue
                break
            else:
                continue

if __name__ == "__main__":
    main()

