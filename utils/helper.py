import os
import logging
import numpy as np
class ModelSaver(object):
    """handle the number of saved models"""
    def __init__(self, max_count):
        self.save_list = []
        self.max_count = max_count

    def append(self, save_path):
        if len(self.save_list) < self.max_count:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)

class RunningAverageTracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.average = 1.0 * self.sum / self.count

    def get_average(self):
        return self.average

    def get_count(self):
        return self.count
    
def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
class ValidationTracker:
    def __init__(self, patience=10, verbose=False, delta=0, save_checkpoint_callback=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_checkpoint_callback (callable): Function to call for saving the model.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.inf
        self.delta = delta
        self.save_checkpoint_callback = save_checkpoint_callback

    def __call__(self, val_score, model, epoch):
        score = -val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, epoch)
            self.counter = 0

    def save_checkpoint(self, val_score, epoch):
        '''Saves model when validation score decreases.'''
        if self.verbose:
            logging.info(f'Validation score improved ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        if self.save_checkpoint_callback:
            self.save_checkpoint_callback(epoch)
        self.val_score_min = val_score