from PIL import Image
import torch.utils.data as data
import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random


class TreeCountingDataset(data.Dataset):
    def __init__(self, root_path, filter_func=None, augment=True):
        """
        Dataset class for tree counting images and corresponding keypoints.

        Args:
        - root_path (str): Path to the directory containing the images and keypoints.
        - filter_func (callable, optional): Function for filtering the dataset. If None, all images are included.
        - augment (bool): Whether to apply data augmentation. Default is True.
        """
        self.root_path = root_path
        self.augment = augment

        all_images = [f for f in os.listdir(root_path) if f.endswith('.png') or f.endswith('.jpg')]
        
        if filter_func:
            self.list_of_images = sorted([f for f in all_images if filter_func(f)])
        else:
            self.list_of_images = sorted(all_images)

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # For RGB images
        ])

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_path, self.list_of_images[index])
        labels_path = image_path.replace('.png', '.npy').replace('.jpg', '.npy')

        image = Image.open(image_path).convert('RGB')
        labels = np.load(labels_path)
        labels = torch.from_numpy(labels).float()

        # Apply random horizontal and vertical flip augmentations during training
        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                if len(labels) > 0:
                    width = image.width
                    labels[:, 0] = width - labels[:, 0]  # Flip x-coordinates
            
            # Vertical flip
            if random.random() > 0.5:
                image = F.vflip(image)
                if len(labels) > 0:
                    height = image.height
                    labels[:, 1] = height - labels[:, 1]  # Flip y-coordinates

        # Apply normalization after augmentation
        image = self.trans(image)
        name = self.list_of_images[index]

        return image, labels, name
