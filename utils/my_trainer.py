from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter, GaussianKernel
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch import optim
import torch.utils.data.dataloader
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.unet import Unet
from models.IndivBlur import IndivBlur
from datasets.tree_counting_dataset import TreeCountingDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    x = torch.stack(transposed_batch[0], 0)
    y = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    return x, y

def load_checkpoint(checkpoint_path):
    suffix = checkpoint_path.rsplit('.', 1)[-1]
    
    if suffix == 'tar':
        checkpoint = torch.load(checkpoint_path, 'cuda' if torch.cuda.is_available() else "cpu")
        model = checkpoint['model_state_dict']
        optimizer = checkpoint['optimizer_state_dict']
        kernel_size = checkpoint['kernel_size']
        start_epoch = checkpoint['epoch']
        refiner = checkpoint.get('refiner_state_dict', None)
        refiner_optimizer = checkpoint.get('refiner_optimizer_state_dict', None)
        return model, refiner, optimizer, refiner_optimizer, kernel_size, start_epoch

class MyTrainer(Trainer):
    def setup(self):
        """Initializes datasets, model, refiner, loss, and optimizer."""
        args = self.args

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
        
        self.datasets = {
            'train': TreeCountingDataset(root_path=os.path.join(args.data_dir, 'train')),
            'val': TreeCountingDataset(root_path=os.path.join(args.data_dir, 'val')),
        }
        
        self.dataloaders = {
            'train': DataLoader(self.datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=train_collate),
            'val': DataLoader(self.datasets['val'], batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=default_collate),
        }
        
        if args.resume:
            model_state_dict, refiner_state_dict, optimizer_state_dict, refiner_optimizer_state_dict, resume_kernel_size, resume_epoch = load_checkpoint(args.resume)
        else:
            model_state_dict = None
            refiner_state_dict = None
            optimizer_state_dict = None
            refiner_optimizer_state_dict = None
            resume_kernel_size = args.kernel_size
            resume_epoch = 0

        self.kernel_size = resume_kernel_size if args.resume else args.kernel_size

        # Model setup
        self.model = Unet()

        # Check if IndivBlur is used
        if args.use_indivblur:
            self.refiner = IndivBlur(kernel_size=self.kernel_size, softmax=args.softmax, downsample=args.downsample)
            self.refiner.to(self.device)
            refiner_params = list(self.refiner.parameters())
            self.refiner_optimizer = optim.Adam(refiner_params, lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.refiner = None
            self.refiner_optimizer = None
            self.kernel_generator = GaussianKernel(kernel_size=self.kernel_size, downsample=args.downsample, device=self.device)

        # Get params from models
        params = list(self.model.parameters())

        # Moving model to the correct device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        # Resume from checkpoint (if applicable)
        self.start_epoch = 0
        if args.resume:
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optimizer_state_dict)
            if args.use_indivblur and refiner_state_dict is not None:
                self.refiner.load_state_dict(refiner_state_dict)
                self.refiner_optimizer.load_state_dict(refiner_optimizer_state_dict)
            self.start_epoch = resume_epoch + 1

        self.criterion = torch.nn.MSELoss(reduction='sum')

        # Saving model checkpoints
        self.list_of_saved_checkpoint_models = Save_Handle(max_num=args.max_model_num)
        self.list_of_saved_valuation_models = Save_Handle(max_num=args.max_model_num)
        
        # Track best model performance during validation
        self.best_mae = np.inf
        self.best_rmse = np.inf
        self.best_loss = np.inf
        self.best_epoch = 0

    def train(self):
        """training process"""
        args = self.args
        
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            
            self.epoch = epoch
            self.train_epoch(epoch)
            
            # Validate if epoch matches the right interval
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self, epoch=0):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()  # To track MAE
        epoch_rmse = AverageMeter()  # To track RMSE
        epoch_start = time.time()
        self.model.train() 
        if self.args.use_indivblur:
            self.refiner.train()    

        # Iterate over data
        for step, (x, y) in enumerate(self.dataloaders['train']):
            tree_count = np.array([len(p) for p in y], dtype=np.float32)  # Ground truth counts
            x = x.to(self.device) # batch images
            y = [p.to(self.device) for p in y] # batch points

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                if self.args.use_indivblur:
                    self.refiner_optimizer.zero_grad()

                # Forward pass through model
                outputs = self.model(x)  # Predict

                # Generate ground truth density maps
                if self.args.use_indivblur:
                    pred = self.refiner(y, x, outputs.shape)  # Refine
                else:
                    pred = self.kernel_generator.generate_density_map(y, outputs.shape)

                # Compute loss (including cos_loss)
                loss = self.criterion(pred, outputs)
                loss += self.args.cos_loss_weight * cos_loss(pred, outputs) 
                loss.backward() 

                # Update optimizers
                self.optimizer.step()
                if self.args.use_indivblur:
                    self.refiner_optimizer.step()

                # Calculate predicted count (sum over spatial dimensions) and difference from ground truth
                pre_count = outputs.sum(dim=(1, 2, 3)).detach().cpu().numpy()  # Sum across spatial dimensions
                difference = pre_count - tree_count  # Calculate error per sample in the batch

                # Update loss, MAE, and RMSE metrics
                N = x.shape[0]
                epoch_loss.update(loss.item(), N)
                epoch_mae.update(np.mean(abs(difference)).item(), N)  # Mean Absolute Error (MAE)
                epoch_rmse.update(np.sqrt(np.mean(difference ** 2)).item(), N)  # Root Mean Square Error (RMSE) 

        # Logging MAE and RMSE
        logging.info(f'Epoch {epoch} training: Loss: {epoch_loss.get_avg():.2f}, RMSE: {epoch_rmse.get_avg():.2f}, MAE: {epoch_mae.get_avg():.2f}, Cost {time.time() - epoch_start:.1f} sec') 

        # Save model only if loss improved
        if epoch_loss.get_avg() < self.best_loss:
            self.best_loss = epoch_loss.get_avg()
            save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.tar')
            self.list_of_saved_checkpoint_models.append(save_path)  # Control the number of saved models
            save_dict = {
                'epoch': epoch,
                'kernel_size': self.kernel_size,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.args.use_indivblur:
                save_dict['refiner_state_dict'] = self.refiner.state_dict()
                save_dict['refiner_optimizer_state_dict'] = self.refiner_optimizer.state_dict()
            torch.save(save_dict, save_path)


    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()
        if self.args.use_indivblur:
            self.refiner.eval()
        epoch_res = []

        for inputs, points in self.dataloaders['val']:
            inputs = inputs.to(self.device)

            # For flexibility, consider if the batch size > 1 is possible or required
            assert inputs.size(0) == 1, 'The batch size should equal to 1 in validation mode'

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                points = points[0].type(torch.LongTensor)
                res = len(points) - torch.sum(outputs).item()
                epoch_res.append(res)

        # Convert to numpy array for metric calculation
        epoch_res = np.array(epoch_res)

        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        # Logging RMSE and MAE
        logging.info('Epoch {} validation, RMSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, rmse, mae, time.time() - epoch_start))

        # If this is the best result so far
        if (rmse + mae) < (self.best_rmse + self.best_mae):
            self.best_rmse = rmse
            self.best_mae = mae
            self.best_epoch = self.epoch

            logging.info("Saving best validation model. RMSE {:.2f} MAE {:.2f} epoch {}".format(self.best_rmse, self.best_mae, self.epoch))

            save_path = os.path.join(self.save_dir, f'best_checkpoint_epoch_{self.epoch}.tar')
            self.list_of_saved_valuation_models.append(save_path)  # Control the number of saved models
            save_dict = {
                'epoch': self.epoch,
                'kernel_size': self.kernel_size,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.args.use_indivblur:
                save_dict['refiner_state_dict'] = self.refiner.state_dict()
                save_dict['refiner_optimizer_state_dict'] = self.refiner_optimizer.state_dict()
            torch.save(save_dict, save_path)


def cos_loss(output, target):
    B = output.shape[0]
    output = output.reshape(B, -1)
    target = target.reshape(B, -1)
    loss = torch.mean(1 - F.cosine_similarity(output, target))
    return loss
