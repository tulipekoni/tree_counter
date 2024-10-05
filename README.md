# Kernel-based Density Map Generation for dense tree counting

This project implements a kernel-based density map generation method for dense object counting, specifically applied to tree counting in aerial images.

## Features

- Utilizes a U-Net architecture for density map prediction
- Implements both Gaussian kernel and IndivBlur methods for ground truth generation
- Supports training, validation, and testing phases
- Includes visualization tools for density maps

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/kernel-based-density-map.git
   cd kernel-based-density-map
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

1. Download the Yosemite Tree Dataset using the following command:

   ```
   gdown --folder https://drive.google.com/drive/folders/1NWAqslICPoTS8OvT8zosI0R7cmsl6x9j -O ./data
   ```

   This command uses the `gdown` tool to download the files from the Google Drive folder to the ./data directory.

2. Install gdown if you haven't already:

   ```
   pip install gdown
   ```

3. Preprocess the dataset:
   ```
   python preprocess_dataset.py --origin_dir ./data --data_dir <path_to_processed_data>
   ```

## Pretrained model

The pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1TaY5I1eHIt7pm2YBfqw4BfnpX2l3Bof4?usp=sharing).

## Training

To train the model, run:

```
python train.py
```

You can override config parameters using command-line arguments:

```
python train.py  --override data_dir ./new_data_path lr 1e-4
```

## Testing

To test the model:

```
python test.py --model_folder ./path_to_model_folder
```

## Configuration

The `config.json` file contains all the hyperparameters and settings for the model. You can modify this file to change the model's behavior. Key parameters include:

## Project Structure

- `models/`: Contains the U-Net and IndivBlur model implementations
- `datasets/`: Includes the TreeCountingDataset class
- `utils/`: Helper functions and classes for training and evaluation
- `train.py`: Main script for training the model
- `test.py`: Script for evaluating the model on the test set
- `test_visualize.py`: Script for visualizing density maps

## Citation

If you use this code in your research, please cite:

```
@article{your-paper,
title={Kernel-based Density Map Generation for Dense Object Counting},
author={Your Name},
journal={Your Journal},
year={2024}
}
```
