import torch
import torch.nn as nn

class StaticRefiner(nn.Module):
    def __init__(self, device, sigma=15):
        super(StaticRefiner, self).__init__()
        self.sigma = sigma
        self.device = device
        # Calculate kernel size based on sigma
        self.kernel_size = self.calculate_kernel_size(sigma)
        
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        self.gaussian_kernel = kernel / torch.sum(kernel)

    def calculate_kernel_size(self, sigma):
        kernel_size = int(6 * sigma) + 3
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def forward(self,batch_images, batch_labels):
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