import os
import torch
from utils.static import Static
from utils.adaptive import Adaptive
from utils.config_loader import load_config, override_config
from utils.arg_parser import parse_train_args, parse_args_to_config

def main():
    args = parse_train_args()
    config = load_config(args.config)

    # Override config with command line arguments
    if args.override:
        overrides = parse_args_to_config(args)
        config = override_config(config, overrides)

    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = config['device'].strip()
    trainer = Adaptive(config)
    trainer.setup()
    trainer.train()

if __name__ == "__main__":
    main()