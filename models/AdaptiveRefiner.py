import torch
import torch.nn as nn
from models.StaticRefiner import StaticRefiner

class AdaptiveRefiner(StaticRefiner):
    def __init__(self, device, config):
        super().__init__(device, sigma=15.0)  # Initialize parent first
        self.sigma_param = nn.Parameter(torch.tensor(15.0, device=device, requires_grad=True))
        self._cached_sigma = None
        self._cached_kernel = None
        
        self.optimizer = torch.optim.Adam([self.sigma_param], lr=config['sigma_lr'])

    def _update_kernel(self):
        if self._cached_sigma != self.sigma_param.item():
            self.kernel_size = self.calculate_kernel_size(self.sigma_param.item())
            ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=self.device)
            xx, yy = torch.meshgrid(ax, ax, indexing='ij')
            kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma_param.item()**2))
            self.gaussian_kernel = kernel / torch.sum(kernel)
            self._cached_sigma = self.sigma_param.item()
            self._cached_kernel = self.gaussian_kernel

    def forward(self, batch_images, batch_labels):
        self._update_kernel()
        return super().forward(batch_images, batch_labels)

    def get_sigma(self):
        return self.sigma_param.item()

    def step(self):
        torch.nn.utils.clip_grad_norm_([self.sigma_param], max_norm=1.0)
        self.optimizer.step()
        
        with torch.no_grad():
            self.sigma_param.clamp_(min=1.0, max=30.0)
        
        self.optimizer.zero_grad()

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        super().eval()
        return self