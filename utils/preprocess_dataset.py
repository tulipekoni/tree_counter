import os
import random
import numpy as np
from PIL import Image
from arg_parser import parse_preprocess_args

Image.MAX_IMAGE_PIXELS = 1262080000

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

    if max_images is not None:
        patches = patches[:max_images]

    random.shuffle(patches)

    # If validation split is needed
    if is_val_split:
        split_index = int(len(patches) * (1 - args.val_split))
        train_patches = patches[:split_index]
        val_patches = patches[split_index:]

        save_patches(train_patches, region_name, save_dir, is_val=False)
        save_patches(val_patches, region_name, save_dir.replace('train', 'val'), is_val=True)
    else:
        save_patches(patches, region_name, save_dir, is_val=False)


def save_patches(patches, region_name, save_dir, is_val=False):
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
        y_coords, x_coords = np.where(label_patch == 255)
        # Stack coordinates in (x,y) order and ensure they're within patch bounds
        object_positions = np.column_stack((x_coords, y_coords))
        
        # Verify coordinates are within patch bounds
        patch_height, patch_width = data_patch.shape[:2]
        mask = (object_positions[:, 0] >= 0) & (object_positions[:, 0] < patch_width) & \
               (object_positions[:, 1] >= 0) & (object_positions[:, 1] < patch_height)
        object_positions = object_positions[mask]

        # Save the object positions as a .npy file
        npy_save_path = patch_save_path.replace('.png', '.npy')
        np.save(npy_save_path, object_positions) 

        patch_count += 1


if __name__ == '__main__':
    args = parse_preprocess_args()

    data_path = args.data_path
    
    data_image = Image.open(data_path + '/z20_data.png')
    label_image = Image.open(data_path + '/z20_label.png')

    data_array = np.array(data_image)
    label_array = np.array(label_image)

    padding = 4000
    
    # Split the images into regions (A, B, C, D)
    data_columns = split_image(data_array, padding)
    label_columns = split_image(label_array, padding)
    
    regions = {'A': 'test', 'B': 'train', 'C': 'test', 'D': 'train'}
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # Process each region and 
    for i, region_name in enumerate(['A', 'B', 'C', 'D']):
        save_dir = os.path.join(args.save_dir, regions[region_name])

        # If region is for training, split part for validation
        is_val_split = regions[region_name] == 'train'
        
        print(f'Processing region {region_name} ({regions[region_name]})...')
        process_and_save(data_columns[i], label_columns[i], region_name, save_dir, patch_size=args.subimage_size, max_images=args.max_images, is_val_split=is_val_split)
    
    print("Processing complete.")