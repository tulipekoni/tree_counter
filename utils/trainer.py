import os
import logging
from datetime import datetime
from utils.logger import setlogger
import json

class Trainer(object):
    def __init__(self, config):
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(config['save_dir'], sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger
        for k, v in config.items():  # log all config key-value pairs
            logging.info("{}: {}".format(k, v))
        
        config_path = os.path.join(self.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass
