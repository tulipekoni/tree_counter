from utils.arg_parser import parse_train_args, parse_args_to_config
from utils.config_loader import load_config, override_config
from utils.my_trainer import MyTrainer
import argparse
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--config', default='config.json', help='path to config file')
    parser.add_argument('--override', nargs='*', help='override config parameters')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_train_args()
    config = load_config(args.config)

    # Override config with command line arguments
    if args.override:
        overrides = parse_args_to_config(args)
        config = override_config(config, overrides)

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = config['device'].strip()  # set vis gpu
    trainer = MyTrainer(config)
    trainer.setup()
    trainer.train()
