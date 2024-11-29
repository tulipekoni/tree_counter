import os
import time
import torch
import logging
from models.DMG import DMG
from utils.trainer import Trainer
import torch.utils.data.dataloader
from torch.optim import lr_scheduler, Adam
from utils.helper import RunningAverageTracker

class Adaptive(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        config = self.config
        self.dmg = DMG(device=self.device, initial_sigma_value=self.sigma, requires_grad=True)
        self.dmg.to(self.device)
        super().setup()
        
        params = list(self.dmg.parameters())
        self.dmg_optimizer = Adam(params, lr=config['dmg_lr'])
        self.dmg_lr_scheduler = lr_scheduler.StepLR(self.dmg_optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

        
    def train_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        self.model.train() 
        self.dmg.train() 

        loss_components_sum = {'pixel_loss': 0, 'count_loss': 0, 'cos_loss': 0, 'sigma_penalty': 0}
        
        # Iterate over data
        for step, (batch_images, batch_labels, batch_names) in enumerate(self.dataloaders['train']):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            with torch.set_grad_enabled(True):
                self.model_optimizer.zero_grad()
                self.dmg_optimizer.zero_grad()
                
                batch_pred_density_maps = self.model(batch_images) 
                batch_gt_density_maps = self.dmg(batch_images, batch_labels)

                # Loss for step
                loss, components = self.loss_function(batch_pred_density_maps, batch_gt_density_maps, self.dmg.sigma)
                loss.backward() 
                
                # Scale DMG sigma gradients
                sigma_scale = 10.0  
                if self.dmg.sigma.grad is not None:
                    self.dmg.sigma.grad *= sigma_scale
                
                # Update component sums
                for k, v in components.items():
                    loss_components_sum[k] += v
                
                self.model_optimizer.step()
                self.dmg_optimizer.step()

                # The number of trees is total sum of all prediction pixels
                batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()  
                batch_differences = batch_pred_counts - batch_gt_count 

                # Update loss, MAE, and RMSE metrics
                batch_size = batch_pred_counts.shape[0]
                epoch_loss.update(loss.item(), batch_size)
                epoch_mae.update(torch.abs(batch_differences).sum().item(), batch_size)
                epoch_rmse.update(torch.sum(batch_differences ** 2).item(), batch_size)


        self.dmg_lr_scheduler.step()
        # Calculate averages
        num_batches = len(self.dataloaders['train'])
        avg_components = {k: v/num_batches for k, v in loss_components_sum.items()}
        total_loss = sum(avg_components.values())
        
        # Calculate percentages
        loss_percentages = {k: (v/total_loss)*100 for k, v in avg_components.items()}
        
        average_loss = epoch_loss.get_average()
        average_mae = epoch_mae.get_average()
        average_rmse = torch.sqrt(torch.tensor(epoch_rmse.get_average())).item()
        logging.info(f'Training: Loss: {average_loss:.2f}, RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, Sigma: {self.dmg.sigma.item():.2f}, Cost {time.time() - start_time:.1f} sec\n'
                    f'Loss Components (%):\n'
                    f'  Pixel Loss: {avg_components["pixel_loss"]:.4f} ({loss_percentages["pixel_loss"]:.1f}%)\n'
                    f'  Count Loss: {avg_components["count_loss"]:.4f} ({loss_percentages["count_loss"]:.1f}%)\n'
                    f'  Cos Loss: {avg_components["cos_loss"]:.4f} ({loss_percentages["cos_loss"]:.1f}%)\n'
                    f'  Sigma Loss: {avg_components["sigma_penalty"]:.4f} ({loss_percentages["sigma_penalty"]:.1f}%)')

        return average_loss, average_rmse, average_mae
    
    def validate_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        self.model.eval()
        self.dmg.eval() 

        for batch_images, batch_labels, batch_names in self.dataloaders['val']:
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            with torch.set_grad_enabled(False):
                batch_pred_density_maps = self.model(batch_images)
                batch_gt_density_maps = self.dmg(batch_images, batch_labels)

                # Compute loss
                loss, _ = self.loss_function(batch_pred_density_maps, batch_gt_density_maps, self.dmg.sigma)

                # The number of trees is total sum of all prediction pixels
                batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()
                batch_differences = batch_pred_counts - batch_gt_count

                # Update loss, MAE, and RMSE metrics
                batch_size = batch_pred_counts.shape[0]
                epoch_loss.update(loss.item(), batch_size)
                epoch_mae.update(torch.abs(batch_differences).sum().item(), batch_size)
                epoch_rmse.update(torch.sum(batch_differences ** 2).item(), batch_size)

        average_loss = epoch_loss.get_average()
        average_mae = epoch_mae.get_average()
        average_rmse = torch.sqrt(torch.tensor(epoch_rmse.get_average())).item()
        logging.info(f'Validation: Loss: {average_loss:.2f}, RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, Cost {time.time() - start_time:.1f} sec')

        return average_loss, average_rmse, average_mae
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'val_maes': self.val_maes,
            'val_rmses': self.val_rmses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'train_rmses': self.train_rmses,
            'train_losses': self.train_losses,
            
            'best_val_mae': self.best_val_mae,
            'best_val_rmse': self.best_val_rmse,
            
            'sigma': self.dmg.sigma.item(),
            'dmg_optimizer_state_dict': self.dmg_optimizer.state_dict(),
            'dmg_lr_scheduler_state_dict': self.dmg_lr_scheduler.state_dict(),
            
            'model_state_dict': self.model.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'model_lr_scheduler_state_dict': self.model_lr_scheduler.state_dict(),
        }
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.tar')
        self.list_of_best_models.append(save_path)
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved!")
        pass

    def load_checkpoint(self):
        config = self.config
        
        # Find the most recent .tar file in the resume folder
        checkpoint_files = [f for f in os.listdir(config['resume']) if f.endswith('.tar')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No .tar checkpoint files found in {config['resume']}")
        
        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(config['resume'], f)))
        checkpoint_path = os.path.join(config['resume'], latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if config['load_weights_only'] is False:
            self.sigma = checkpoint['sigma']
            self.start_epoch = checkpoint['epoch'] + 1
            self.val_maes = checkpoint['val_maes']
            self.val_rmses = checkpoint['val_rmses']
            self.val_losses = checkpoint['val_losses']
            self.train_maes = checkpoint['train_maes']
            self.train_rmses = checkpoint['train_rmses']
            self.train_losses = checkpoint['train_losses']

            self.best_val_rmse = checkpoint['best_val_rmse']
            self.best_val_mae = checkpoint['best_val_mae']

            self.dmg_optimizer.load_state_dict(checkpoint['dmg_optimizer_state_dict'])
            self.dmg_lr_scheduler.load_state_dict(checkpoint['dmg_lr_scheduler_state_dict'])
            self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
            self.model_lr_scheduler.load_state_dict(checkpoint['model_lr_scheduler_state_dict'])
            self.dmg.sigma.data = torch.tensor(self.sigma, device=self.device)
            logging.info(f"Checkpoint loaded!")
        else:
            logging.info(f"Model weights loaded!")


