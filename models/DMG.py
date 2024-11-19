import torch
import torch.nn as nn

class DMG(nn.Module):
    def __init__(self, device, initial_sigma_value, requires_grad=True):
        super(DMG, self).__init__()
        self.device = device
        self.sigma = nn.Parameter(torch.tensor(initial_sigma_value, dtype=torch.float32, device=device), requires_grad=requires_grad)
        self.kernel_size = self.calculate_kernel_size(self.sigma)
        self.gaussian_kernel = self.calculate_gaussian_kernel()

    def calculate_kernel_size(self, sigma):
        kernel_size = torch.ceil(6 * sigma + 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return int(kernel_size.item())

    def calculate_gaussian_kernel(self):
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma.pow(2)))
        return kernel / torch.sum(kernel)

    def forward(self, batch_images, batch_labels):
        shape = batch_images.shape
        density = torch.zeros((shape[0], 1, shape[2], shape[3]), device=self.device)
        
        for batch, labels in enumerate(batch_labels):
            if len(labels) == 0:
                continue
            
            for label in labels:
                # Create coordinate grids centered at each point
                x, y = label[0], label[1]
                x_grid = torch.arange(0, shape[2], device=self.device).float()
                y_grid = torch.arange(0, shape[3], device=self.device).float()
                y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
                
                # Calculate Gaussian in a differentiable way
                gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * self.sigma**2))
                gaussian = gaussian / gaussian.sum()
                
                density[batch, 0] += gaussian

        return density
