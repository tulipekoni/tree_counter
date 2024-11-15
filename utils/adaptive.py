import os
import time
import torch
import logging
import numpy as np
import torch.utils.data.dataloader
from utils.trainer import Trainer
from utils.helper import RunningAverageTracker
from models.AdaptiveRefiner import AdaptiveRefiner

class Adaptive(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        super().setup()
        self.refiner = AdaptiveRefiner(device=self.device)
        self.refiner.to(self.device)
        
        # Initialize optimizer with both model and refiner parameters
        params = list(self.model.parameters()) + list(self.refiner.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config['lr'])
        
        # Scheduler setup
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['lr_step_size'], 
            gamma=self.config['lr_gamma']
        )

    def train_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        
        # Set train mode
        self.model.train()
        self.refiner.train()

        # Iterate over data
        for step, (batch_images, batch_labels, batch_names) in enumerate(self.dataloaders['train']):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            with torch.set_grad_enabled(True):
                # Zero all gradients
                self.optimizer.zero_grad()

                # Forward pass
                batch_pred_density_maps = self.model(batch_images)
                batch_gt_density_maps = self.refiner(batch_images, batch_labels)

                # Calculate loss
                loss = self.loss_function(batch_pred_density_maps, batch_gt_density_maps)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()

                # Calculate metrics
                batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()
                batch_differences = batch_pred_counts - batch_gt_count

                # Update metrics
                batch_size = batch_pred_counts.shape[0]
                epoch_loss.update(loss.item(), batch_size)
                epoch_mae.update(torch.abs(batch_differences).sum().item(), batch_size)
                epoch_rmse.update(torch.sum(batch_differences ** 2).item(), batch_size)

        average_loss = epoch_loss.get_average()
        average_mae = epoch_mae.get_average()
        average_rmse = torch.sqrt(torch.tensor(epoch_rmse.get_average())).item()
        
        logging.info(f'Training: Loss: {average_loss:.2f}, RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, '
                    f'Sigma: {self.refiner.get_sigma():.2f}, Cost {time.time() - start_time:.1f} sec')

        return average_loss, average_rmse, average_mae

    def validate_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        
        # Set eval mode
        self.model.eval()
        self.refiner.eval()

        for batch_images, batch_labels, batch_names in self.dataloaders['val']:
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            with torch.set_grad_enabled(False):
                batch_pred_density_maps = self.model(batch_images)
                batch_gt_density_maps = self.refiner(batch_images, batch_labels)

                # Compute loss
                loss = self.loss_function(batch_pred_density_maps, batch_gt_density_maps)

                # Calculate predicted counts and differences
                batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()
                batch_differences = batch_pred_counts - batch_gt_count

                # Update metrics
                batch_size = batch_pred_counts.shape[0]
                epoch_loss.update(loss.item(), batch_size)
                epoch_mae.update(torch.abs(batch_differences).sum().item(), batch_size)
                epoch_rmse.update(torch.sum(batch_differences ** 2).item(), batch_size)

        average_loss = epoch_loss.get_average()
        average_mae = epoch_mae.get_average()
        average_rmse = torch.sqrt(torch.tensor(epoch_rmse.get_average())).item()
        logging.info(f'Validation: Loss: {average_loss:.2f}, RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, '
                    f'Sigma: {self.refiner.get_sigma():.2f}, Cost {time.time() - start_time:.1f} sec')

        return average_loss, average_rmse, average_mae

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'sigma': self.refiner.get_sigma(),
            'val_maes': self.val_maes,
            'val_rmses': self.val_rmses,
            'val_losses': self.val_losses,
            'train_maes': self.train_maes,
            'train_rmses': self.train_rmses,
            'train_losses': self.train_losses,
            
            'best_val_mae': self.best_val_mae,
            'best_val_rmse': self.best_val_rmse,
            
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'sigma_state_dict': self.refiner.sigma_param,
        }
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.tar')
        self.list_of_best_models.append(save_path)
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved! Current sigma: {self.refiner.get_sigma():.2f}")

    def load_checkpoint(self):
        config = self.config
        
        if config['resume']:
            # Full training resume - load everything
            checkpoint_files = [f for f in os.listdir(config['resume']) if f.endswith('.tar')]
            if not checkpoint_files:
                raise FileNotFoundError(f"No .tar checkpoint files found in {config['resume']}")
            
            latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(config['resume'], f)))
            checkpoint_path = os.path.join(config['resume'], latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load all training state
            self.refiner.sigma_param.data = checkpoint['sigma_state_dict'].to(self.device)
            self.start_epoch = checkpoint['epoch'] + 1
            self.val_maes = checkpoint['val_maes']
            self.val_rmses = checkpoint['val_rmses']
            self.val_losses = checkpoint['val_losses']
            self.train_maes = checkpoint['train_maes']
            self.train_rmses = checkpoint['train_rmses']
            self.train_losses = checkpoint['train_losses']
            
            self.best_val_rmse = checkpoint['best_val_rmse']
            self.best_val_mae = checkpoint['best_val_mae']
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            logging.info(f"Training resumed from checkpoint! Current sigma: {self.refiner.get_sigma():.2f}")
            
        elif config['model_dir']:
            # Only load model weights - for inference or fresh training from pretrained
            checkpoint_files = [f for f in os.listdir(config['model_dir']) if f.endswith('.tar')]
            if not checkpoint_files:
                raise FileNotFoundError(f"No .tar checkpoint files found in {config['model_dir']}")
            
            latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(config['model_dir'], f)))
            checkpoint_path = os.path.join(config['model_dir'], latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Only load model and optimizer weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logging.info(f"Model weights loaded from checkpoint! Starting fresh training.")