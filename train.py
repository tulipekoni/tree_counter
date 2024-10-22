import os
import torch
from utils.static import Static
from utils.arg_parser import parse_train_args, parse_args_to_config
from utils.config_loader import load_config, override_config

if __name__ == '__main__':
    args = parse_train_args()
    config = load_config(args.config)

    # Override config with command line arguments
    if args.override:
        overrides = parse_args_to_config(args)
        config = override_config(config, overrides)

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = config['device'].strip()  # set vis gpu
    trainer = Static(config)
    trainer.setup()
    trainer.train()
