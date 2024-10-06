import argparse
from typing import Dict, Any

def parse_train_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train Tree Counting Model')
    parser.add_argument('--config', default='config.json', help='path to config file')
    parser.add_argument('--override', nargs='*', help='override config parameters')
    return parser.parse_args()

def parse_test_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the testing script.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Test Tree Counting Model')
    parser.add_argument('--model_folder', type=str, required=True, help='Path to the folder containing the model checkpoint and config')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory containing the test data')
    return parser.parse_args()

def parse_preprocess_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Preprocess Tree Counting Dataset')
    parser.add_argument('--data_path', type=str, default='./data', required=False, help='Path to the original dataset.')
    parser.add_argument('--save_dir', type=str, default='./processed_data', required=False, help='Directory to save processed data.')
    parser.add_argument('--block_size', type=int, default=320, help='Size of image patches.')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process (for debugging).')
    parser.add_argument('--val-split', type=float, default=0.2, help='Percentage of the training data to use for validation.')
    return parser.parse_args()

def parse_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert parsed arguments to a configuration dictionary.

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = vars(args)
    if 'override' in config and config['override']:
        overrides = {}
        for override in config['override']:
            key, value = override.split('=')
            overrides[key] = value
        config.update(overrides)
        del config['override']
    return config
