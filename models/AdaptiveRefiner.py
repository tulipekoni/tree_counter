import torch
import torch.nn as nn

class AdaptiveRefiner(nn.Module):
    def __init__(self, device, initial_sigma=15.0):
        super(AdaptiveRefiner, self).__init__()
        self.device = device
        self.sigma = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32, device=device))
        
    def calculate_kernel_size(self, sigma):
        kernel_size = int(6 * sigma) + 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        return min(kernel_size, 35*6+3)

    def get_gaussian_kernel(self):
        sigma = torch.abs(self.sigma)  # Ensure sigma is positive
        kernel_size = self.calculate_kernel_size(sigma)
        
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / torch.sum(kernel), kernel_size

    def forward(self, batch_images, batch_labels):
        gaussian_kernel, kernel_size = self.get_gaussian_kernel()
        
        # Create an empty density map for this image
        shape = batch_images.shape
        padded_shape = (shape[0], 1, shape[2] + 2 * kernel_size, shape[3] + 2 * kernel_size)
        density = torch.zeros(padded_shape, device=self.device)
        
        # For each batch...
        for batch, labels in enumerate(batch_labels):
            if len(labels) == 0:
                continue
            
            # For each label
            for label in labels:
                x = int(label[0] - kernel_size/2) + kernel_size
                y = int(label[1] - kernel_size/2) + kernel_size
                xmax = x + kernel_size
                ymax = y + kernel_size

                density[batch, :, x:xmax, y:ymax] += gaussian_kernel
        
        # Remove padding from each batch density map    
        density = density[:, :, kernel_size:-kernel_size, kernel_size:-kernel_size]
        return density
