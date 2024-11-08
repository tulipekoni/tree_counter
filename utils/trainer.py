import os
import json
import torch
import logging
import numpy as np
from models.UNet import UNet
from datetime import datetime
import matplotlib.pyplot as plt
import torch.utils.data.dataloader
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from utils.helper import ModelSaver, setlogger
from torch.utils.data.dataloader import default_collate
from datasets.tree_counting_dataset import TreeCountingDataset
import torch.nn.functional as F
from torch.nn import MSELoss
from utils.losses import combined_loss


class Trainer(ABC):
    def __init__(self, config):
        self.config = config
        
        self.train_maes = []
        self.train_rmses = []
        self.train_losses = []
        
        self.val_maes = []
        self.val_rmses = []
        self.val_losses = []

        # Create directory to save stuff during training 
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(config['save_dir'], sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Set logger
        setlogger(os.path.join(self.save_dir, 'train.log')) 
        for key, value in config.items():  # log all config key-value pairs
            logging.info("{}: {}".format(key, value))
        
        # Save config to model directory
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def setup(self):
        """
        Initializes datasets, model, refiner, loss, and optimizer.      
        This method sets up the training environment, including:
        - Setting the device (GPU or CPU)
        - Loading and preparing datasets
        - Initializing the model, refiner (if used), and optimizers
        - Setting up the loss function
        """
        config = self.config
        
        # Setup device (GPU or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1  # Single GPU version
            logging.info(f'Using {self.device_count} GPU(s)')
        else:
            self.device = torch.device('cpu')
            self.device_count = 1
            logging.info('Using CPU')
        pass
    
        self.datasets = {
            'train': TreeCountingDataset(root_path=os.path.join(config['data_dir'], 'train')),
            'val': TreeCountingDataset(root_path=os.path.join(config['data_dir'], 'val')),
        }
        self.dataloaders = {
            'train': DataLoader(
                self.datasets['train'], 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=config['num_workers'], 
                pin_memory=True, 
                collate_fn=self._batch_collate
            ),
            'val': DataLoader(
                self.datasets['val'], 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers'], 
                pin_memory=True,
                collate_fn=self._batch_collate
            ),
        }
        
        # Model setup
        self.model = UNet()
        self.model.to(self.device)
        
        self.loss_function = combined_loss
                
        # Optimizer setup
        params = list(self.model.parameters())
        self.optimizer = Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
        
        # Scheduler setup
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

        self.list_of_best_models = ModelSaver(max_count=config['max_saved_model_count'])
        
        self.start_epoch = 0        
        self.best_val_mae = np.inf
        self.best_val_rmse = np.inf
        
        self.graph, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Load checkpoint if we are continuing training
        if config['resume']:
            self.load_checkpoint()
            self._update_graph(self.start_epoch-1)       
    
    def train(self):
        config = self.config        
        
        for epoch in range(self.start_epoch, config['max_epoch']):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, config['max_epoch'] - 1) + '-'*5)
            
            train_loss, train_rmse, train_mae = self.train_epoch(epoch)
            val_loss, val_rmse, val_mae = self.validate_epoch(epoch)
            
            self.train_maes.append(train_mae)
            self.train_rmses.append(train_rmse)
            self.train_losses.append(train_loss)
            self.val_maes.append(val_mae)
            self.val_rmses.append(val_rmse)
            self.val_losses.append(val_loss)

            self._update_graph(epoch)
            
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]
            logging.info(f'Current learning rate: {current_lr}')
            
            # If this validation was the best one thus far -> save the model on disk
            if(val_rmse + val_mae) < (self.best_val_rmse + self.best_val_mae):
                self.best_val_rmse = val_rmse
                self.best_val_mae = val_mae
                self.save_checkpoint(epoch=epoch)
    
    def _update_graph(self, epoch):
        epochs = range(0, epoch + 1)
        
        # Update loss graph
        self.ax1.clear()
        self.ax1.semilogy(epochs, self.train_losses, label='Train Loss')
        self.ax1.semilogy(epochs, self.val_losses, label='Val Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss (log scale)')
        self.ax1.legend()
        self.ax1.set_title('Training and Validation Loss')

        # Update RMSE graph
        self.ax2.clear()
        self.ax2.semilogy(epochs, self.train_rmses, label='Train RMSE')
        self.ax2.semilogy(epochs, self.val_rmses, label='Val RMSE')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('RMSE (log scale)')
        self.ax2.legend()
        self.ax2.set_title('Training and Validation RMSE')

        # Update MAE graph
        self.ax3.clear()
        self.ax3.semilogy(epochs, self.train_maes, label='Train MAE')
        self.ax3.semilogy(epochs, self.val_maes, label='Val MAE')
        self.ax3.set_xlabel('Epoch')
        self.ax3.set_ylabel('MAE (log scale)')
        self.ax3.legend()
        self.ax3.set_title('Training and Validation MAE')

        self.graph.tight_layout()
        self.graph.savefig(os.path.join(self.save_dir, 'training_graphs.png'))
    
    
    @abstractmethod
    def save_checkpoint(self, epoch):
        pass        
    
    @abstractmethod
    def load_checkpoint(self):
        pass
    
    @abstractmethod
    def train_epoch(self, epoch):
        pass
    
    @abstractmethod
    def validate_epoch(self, epoch):
        pass
    

    @staticmethod
    def _batch_collate(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        labels = transposed_batch[1] 
        path = transposed_batch[2]
        return images, labels, path

