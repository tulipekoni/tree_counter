import torch
import torch.nn as nn

class StaticRefinerTuner(nn.Module):
    def __init__(self, device, sigma):
        super(StaticRefinerTuner, self).__init__()
        self.sigma = sigma
        self.device = device
        self.kernel_size = self.calculate_kernel_size(self.sigma.item())
        self.gaussian_kernel = self.calculate_gaussian_kernel()

    def calculate_kernel_size(self, sigma):
        kernel_size = int(6 * sigma) + 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def calculate_gaussian_kernel(self):
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        return kernel / torch.sum(kernel)

    def forward(self, batch_images, batch_labels):
        # Recalculate kernel size and kernel if sigma has changed
        if self.sigma.requires_grad:
            self.kernel_size = self.calculate_kernel_size(self.sigma.item())
            self.gaussian_kernel = self.calculate_gaussian_kernel()

        # Create an empty density map for this image
        shape = batch_images.shape
        padded_shape = (shape[0], 1, shape[2] + 2 * self.kernel_size, shape[3] + 2 * self.kernel_size)
        density = torch.zeros(padded_shape, device=self.device) 
        
        # For each batch...
        for batch, labels in enumerate(batch_labels):
            if len(labels) == 0:
                continue
            
            # For each label
            for label in labels:
                x = int(label[0] - self.kernel_size/2) + self.kernel_size
                y = int(label[1] - self.kernel_size/2) + self.kernel_size
                xmax = x + self.kernel_size
                ymax = y + self.kernel_size

                density[batch, :, x:xmax, y:ymax] += self.gaussian_kernel
        
        # Remove padding from each batch density map    
        density = density[:, :, self.kernel_size:-self.kernel_size, self.kernel_size:-self.kernel_size]
        return density
