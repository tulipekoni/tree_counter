import PIL
from PIL import Image
import numpy as np
import os
import argparse
PIL.Image.MAX_IMAGE_PIXELS = 1262080000


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess large z20 images and generate .mat files for annotations.')
    parser.add_argument('--data-path', default='z20_data.png', help='Path to the large data image (both data and label should be in the same folder).')
    parser.add_argument('--save-dir', default='./processed_data', help='Directory to save processed patches and .mat files.')
    parser.add_argument('--max-images', type=int, default=None, help='Maximum number of images to process per region. Default is all.')
    parser.add_argument('--block-size', type=int, default=240, help='Size of the training images.')
    parser.add_argument('--val-split', type=float, default=0.2, help='Percentage of the training data to use for validation.')
    return parser.parse_args()


def split_image(image_array, padding, cols=4):
    """
    Splits the image into 4 vertical columns (A, B, C, D) after removing padding.
    """
    height, width = image_array.shape[:2]
    
    # Remove padding
    image_array = image_array[padding:height-padding, padding:width-padding]
    height, width = image_array.shape[:2]
    
    # Split the image into 4 columns
    column_width = width // cols
    return [image_array[:, i*column_width:(i+1)*column_width] for i in range(cols)]


def process_and_save(image_column, label_column, region_name, save_dir, patch_size=240, max_images=None, is_val_split=False):
    """
    Splits the image column into patches, processes the corresponding label patches, and saves them as .mat files.
    """
    height, width = image_column.shape[:2]
    patches = []

    # Iterate through the image in patch_size chunks
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Extract patch from the data and label images
            data_patch = image_column[y:y+patch_size, x:x+patch_size]
            label_patch = label_column[y:y+patch_size, x:x+patch_size]
            patches.append((data_patch, label_patch))

    # Limit the number of patches if max_images is specified
    if max_images is not None:
        patches = patches[:max_images]

    # If validation split is needed
    if is_val_split:
        split_index = int(len(patches) * (1 - args.val_split))
        train_patches = patches[:split_index]
        val_patches = patches[split_index:]

        # Save training patches
        save_patches(train_patches, region_name, save_dir, is_val=False)

        # Save validation patches
        save_patches(val_patches, region_name, save_dir.replace('train', 'val'), is_val=True)
    else:
        # Save all patches (for test regions or regions without validation)
        save_patches(patches, region_name, save_dir, is_val=False)


def save_patches(patches, region_name, save_dir, is_val=False):
    """Helper function to save patches."""
    patch_count = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data_patch, label_patch in patches:
        if save_dir.endswith('/test'):
            patch_type = 'test'
        else:
            patch_type = 'val' if is_val else 'train'
        
        data_patch_image = Image.fromarray(data_patch)
        patch_save_path = os.path.join(save_dir, f'{region_name}_{patch_type}_patch_{patch_count}.png')
        data_patch_image.save(patch_save_path)
        
        # Find white pixels in the label (object positions)
        object_positions = np.column_stack(np.where(label_patch == 255))

        # Save the object positions as a .npy file
        npy_save_path = patch_save_path.replace('.png', '.npy')
        np.save(npy_save_path, object_positions) 

        # Recreate image from object positions
        object_image = np.zeros_like(label_patch)
        object_image[object_positions[:, 0], object_positions[:, 1]] = 255  

        # Save the object position image
        object_image_save_path = os.path.join(args.save_dir + '/points', f'{region_name}_{patch_type}_patch_{patch_count}.png')
        object_image_pil = Image.fromarray(object_image)
        object_image_pil.save(object_image_save_path)

        patch_count += 1


if __name__ == '__main__':
    args = parse_args()

    # Load the large images
    data_path = args.data_path
    
    data_image = Image.open(data_path + '/z20_data.png')
    label_image = Image.open(data_path + '/z20_label.png')

    # Convert to numpy arrays
    data_array = np.array(data_image)
    label_array = np.array(label_image)

    # Define padding size
    padding = 4000
    
    # Split the images into regions (A, B, C, D)
    data_columns = split_image(data_array, padding)
    label_columns = split_image(label_array, padding)
    
    # Define regions and their use for training, testing, and validation
    regions = {'A': 'test', 'B': 'train', 'C': 'test', 'D': 'train'}
    
    # Ensure save directory exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    if not os.path.exists(args.save_dir + '/points'):
        os.makedirs(args.save_dir + '/points')

    # Process each region and save patches and .mat files
    for i, region_name in enumerate(['A', 'B', 'C', 'D']):
        save_dir = os.path.join(args.save_dir, regions[region_name])

        # If region is for training, split part for validation
        is_val_split = regions[region_name] == 'train'
        
        print(f'Processing region {region_name} ({regions[region_name]})...')
        process_and_save(data_columns[i], label_columns[i], region_name, save_dir, patch_size=args.block_size, max_images=args.max_images, is_val_split=is_val_split)
    
    print("Processing complete.")