import os
import time
import torch
import logging
import numpy as np
import torch.utils.data.dataloader
from utils.trainer import Trainer
from utils.helper import RunningAverageTracker
from models.StaticRefiner import StaticRefiner

class Static(Trainer):
    def __init__(self, config):
        super().__init__(config)

    def setup(self):
        super().setup()
        self.refiner = StaticRefiner(device=self.device, sigma=self.sigma)
        self.refiner.to(self.device)
        
    def train_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        self.model.train() 
        # No need to train refiner here since it returns just static kernels

        # Iterate over data
        for step, (batch_images, batch_labels, batch_names) in enumerate(self.dataloaders['train']):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                batch_pred_density_maps = self.model(batch_images) 
                batch_gt_density_maps = self.refiner(batch_images, batch_labels)

                # Loss for step
                loss = self.criterion(batch_pred_density_maps, batch_gt_density_maps)
                loss.backward() 
                self.optimizer.step()

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
        logging.info(f'Training: Loss: {average_loss:.2f}, RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, Cost {time.time() - start_time:.1f} sec') 

        return average_loss, average_rmse, average_mae
    
    def validate_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        self.model.eval()

        logging.info(f"Starting validation for epoch {epoch}")

        for batch_idx, (batch_images, batch_labels, batch_names) in enumerate(self.dataloaders['val']):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            logging.info(f"Batch {batch_idx}: Input image stats - Mean: {batch_images.mean():.4f}, Std: {batch_images.std():.4f}")
            logging.info(f"Batch {batch_idx}: Ground truth counts: {batch_gt_count.tolist()}")

            with torch.set_grad_enabled(False):
                batch_pred_density_maps = self.model(batch_images)
                batch_gt_density_maps = self.refiner(batch_images, batch_labels)

                logging.info(f"Batch {batch_idx}: GT density map stats - Mean: {batch_gt_density_maps.mean():.4f}, Std: {batch_gt_density_maps.std():.4f}, Sum: {batch_gt_density_maps.sum():.4f}")
                logging.info(f"Batch {batch_idx}: Pred density map stats - Mean: {batch_pred_density_maps.mean():.4f}, Std: {batch_pred_density_maps.std():.4f}, Sum: {batch_pred_density_maps.sum():.4f}")

                # Compute loss
                loss = self.criterion(batch_pred_density_maps, batch_gt_density_maps)

                # The number of trees is total sum of all prediction pixels
                batch_pred_counts = batch_pred_density_maps.sum(dim=(1, 2, 3)).detach()
                batch_differences = batch_pred_counts - batch_gt_count

                logging.info(f"Batch {batch_idx}: Predicted counts: {batch_pred_counts.tolist()}")
                logging.info(f"Batch {batch_idx}: Count differences: {batch_differences.tolist()}")
                logging.info(f"Batch {batch_idx}: Batch loss: {loss.item():.4f}")

                # Update loss, MAE, and RMSE metrics
                batch_size = batch_pred_counts.shape[0]
                epoch_loss.update(loss.item(), batch_size)
                epoch_mae.update(torch.abs(batch_differences).sum().item(), batch_size)
                epoch_rmse.update(torch.sum(batch_differences ** 2).item(), batch_size)

            if batch_idx < 5:  # Log detailed info for first 5 batches
                for i in range(batch_size):
                    logging.info(f"Sample {i} in batch {batch_idx}:")
                    logging.info(f"  Image name: {batch_names[i]}")
                    logging.info(f"  GT count: {batch_gt_count[i].item():.2f}")
                    logging.info(f"  Pred count: {batch_pred_counts[i].item():.2f}")
                    logging.info(f"  Difference: {batch_differences[i].item():.2f}")
                    sample_loss = self.criterion(batch_pred_density_maps[i:i+1], batch_gt_density_maps[i:i+1]).item()
                    logging.info(f"  Sample loss: {sample_loss:.4f}")

        average_loss = epoch_loss.get_average()
        average_mae = epoch_mae.get_average()
        average_rmse = torch.sqrt(torch.tensor(epoch_rmse.get_average())).item()
        logging.info(f'Validation: Loss: {average_loss:.4f}, RMSE: {average_rmse:.4f}, MAE: {average_mae:.4f}, Cost {time.time() - start_time:.1f} sec')

        return average_loss, average_rmse, average_mae
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'sigma': self.sigma,
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
        
        self.sigma = checkpoint['sigma'] + 1
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

        logging.info(f"Checkpoint loaded!")
