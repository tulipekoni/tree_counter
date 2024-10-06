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
        img_path = os.path.join(self.root_path, self.list_of_images[index])
        points_path = img_path.replace('.png', '.npy').replace('.jpg', '.npy')

        x = self.trans(Image.open(img_path).convert('RGB'))
        y = np.load(points_path)

        # Return image, keypoints (converted to torch Tensor), and filename
        return x, torch.from_numpy(y).float(), self.list_of_images[index]
