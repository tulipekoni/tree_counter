from utils.my_trainer import MyTrainer 
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--save-dir', default='./checkpoints', 
                        help='directory to save models.')
    parser.add_argument('--data-dir', default='./processed_data',
                        help='training data directory')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='the size of the density map kernel')
    parser.add_argument('--softmax', type=bool, default=False,
                        help='use softmax')
    parser.add_argument('--lr', type=float, default=5e-7,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models to save')
    parser.add_argument('--max-epoch', type=int, default=1000,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=0,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the num of training process')
    parser.add_argument('--downsample', type=int, default=1,
                        help='How much the models output density map is downsampled')
    parser.add_argument('--cos-loss-weight', type=int, default=10,
                        help='cos-loss multiplier')
    parser.add_argument('--use-indivblur', type=bool, default=False,
                        help='Use IndivBlur refiner')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = MyTrainer(args)
    trainer.setup()
    trainer.train()
