import os
import time
import torch
import logging
from utils.trainer import Trainer
from utils.helper import RunningAverageTracker
from torch.optim import Adam, lr_scheduler
from models.StaticRefinerTuner import StaticRefinerTuner

class Tuner(Trainer):
    def __init__(self, config):
        self.train_sigmas = []
        super().__init__(config)

    def setup(self):
        config = self.config
        # Initialize StaticRefinerTuner with initial sigma value
        initial_sigma_value = 3.0  # or retrieve from config if needed
        self.refiner = StaticRefinerTuner(device=self.device, initial_sigma_value=initial_sigma_value)
        self.refiner.to(self.device)
        
        # Call the setup method of the parent class
        super().setup()

        # Define parameter groups with different learning rates
        model_params = list(self.model.parameters())
        refiner_params = list(self.refiner.parameters())

        self.optimizer = Adam([
            {'params': model_params, 'lr': config['lr']},
            {'params': refiner_params, 'lr': config['refiner_lr']}
        ], weight_decay=config['weight_decay'])

        # Scheduler setup
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    def train_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
        self.model.train()
        self.refiner.train()

        for step, (batch_images, batch_labels, batch_names) in enumerate(self.dataloaders['train']):
            batch_gt_count = torch.tensor([len(p) for p in batch_labels], dtype=torch.float32, device=self.device)
            batch_images = batch_images.to(self.device)
            batch_labels = [p.to(self.device) for p in batch_labels]

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                batch_pred_density_maps = self.model(batch_images)
                batch_gt_density_maps = self.refiner(batch_images, batch_labels)

                # Loss for step
                loss = self.loss_function(batch_pred_density_maps, batch_gt_density_maps)
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
        logging.info(f'Training: Loss: {average_loss:.2f}, RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, Sigma: {self.refiner.sigma.item():.2f}, Cost {time.time() - start_time:.1f} sec')

        self.train_sigmas.append(self.refiner.sigma.item())
        return average_loss, average_rmse, average_mae

    def validate_epoch(self, epoch):
        epoch_loss = RunningAverageTracker()
        epoch_mae = RunningAverageTracker()
        epoch_rmse = RunningAverageTracker()
        start_time = time.time()
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
            'train_sigmas': self.train_sigmas,
            'best_val_mae': self.best_val_mae,
            'best_val_rmse': self.best_val_rmse,
            'model_state_dict': self.model.state_dict(),
            'refiner_state_dict': self.refiner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.tar')
        self.list_of_best_models.append(save_path)
        torch.save(checkpoint, save_path)
        logging.info(f"Checkpoint saved!")

    def load_checkpoint(self):
        config = self.config

        # Find the most recent .tar file in the resume folder
        checkpoint_files = [f for f in os.listdir(config['resume']) if f.endswith('.tar')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No .tar checkpoint files found in {config['resume']}")

        latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(config['resume'], f)))
        checkpoint_path = os.path.join(config['resume'], latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if config.get('load_weights_only', False):
            # Only load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'refiner_state_dict' in checkpoint:
                self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
            logging.info(f"Model weights loaded!")
        else:
            # Load full checkpoint
            self.start_epoch = checkpoint['epoch'] + 1
            self.val_maes = checkpoint['val_maes']
            self.val_rmses = checkpoint['val_rmses']
            self.val_losses = checkpoint['val_losses']
            self.train_maes = checkpoint['train_maes']
            self.train_rmses = checkpoint['train_rmses']
            self.train_losses = checkpoint['train_losses']
            self.train_sigmas = checkpoint['train_sigmas']

            self.best_val_rmse = checkpoint['best_val_rmse']
            self.best_val_mae = checkpoint['best_val_mae']

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.refiner.load_state_dict(checkpoint['refiner_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logging.info(f"Full checkpoint loaded!")
