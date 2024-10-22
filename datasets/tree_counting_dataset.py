from PIL import Image
import torch.utils.data as data
import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms


class TreeCountingDataset(data.Dataset):
    def __init__(self, root_path):
        """
        Dataset class for tree counting images and corresponding keypoints.

        Args:
        - root_path (str): Path to the directory containing the images and keypoints.
        """

        self.root_path = root_path

        self.list_of_images = sorted([f for f in os.listdir(self.root_path) if f.endswith('.png') or f.endswith('.jpg')])

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # For RGB images
        ])

    def __len__(self):
        return len(self.list_of_images)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_path, self.list_of_images[index])
        labels_path = image_path.replace('.png', '.npy').replace('.jpg', '.npy')

        image = self.trans(Image.open(image_path).convert('RGB'))
        labels = torch.from_numpy(np.load(labels_path)).float() 
        name = self.list_of_images[index]

  
        return image, labels , name
