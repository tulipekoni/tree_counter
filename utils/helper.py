import os
import torch

class Save_Handle(object):
    """handle the number of """
    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

class GaussianKernel(object):
    def __init__(self, kernel_size, downsample, device, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma  # Use the sigma from config
        self.downsample = downsample
        self.device = device
        
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * self.sigma**2))
        self.gaussian_kernel = kernel / torch.sum(kernel)
     
    def generate_density_map(self, batch_points, shape):
        # Create an empty density map for this image
        padded_shape = (shape[0], shape[1], shape[2] + 2 * self.kernel_size, shape[3] + 2 * self.kernel_size)
        density = torch.zeros(padded_shape, device=self.device)            # For each point, place a Gaussian kernel
          
        for j, points in enumerate(batch_points):
            num_points = len(points)
            if num_points == 0:
                continue
                
            for point in points:
                x = int(point[0] / self.downsample - self.kernel_size/2) + self.kernel_size
                y = int(point[1] / self.downsample - self.kernel_size/2) + self.kernel_size
                xmax = x + self.kernel_size
                ymax = y + self.kernel_size

                density[j, :, x:xmax, y:ymax] += self.gaussian_kernel
                
        density = density[:, :, self.kernel_size:-self.kernel_size, self.kernel_size:-self.kernel_size]
        return density