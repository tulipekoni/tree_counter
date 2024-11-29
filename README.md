# Detecting and counting trees from satellite images

This project implements density map based method for dense object counting, specifically applied to tree counting in aerial images.

## Features

- Utilizes a U-Net architecture for density map prediction
- Implements Density map generation (DMG) model for ground truth generation
- Supports training, validation, and testing phases
- Includes visualization tools for density maps

## Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/kernel-based-density-map.git
   cd kernel-based-density-map
   ```

2. Run the setup script:

   ```bash
   sh ./startup.sh
   ```

   This script will:

   - Create and activate a virtual environment
   - Install all required packages
   - Download the Yosemite Tree Dataset
   - Preprocess the dataset

## Pretrained model

The pretrained models for each phase referenced in the masters thesis can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1TaY5I1eHIt7pm2YBfqw4BfnpX2l3Bof4?usp=sharing).

## Training

To train the model, run:

```bash
python train.py
```

You can override config parameters using command-line arguments:

```bash
python train.py  --override data_dir=./new_data_path lr=1e-4
```

## Testing

To test the model:

```bash
python test.py --model_folder ./path_to_model_folder
```

## Configuration

The `config.json` file contains all the hyperparameters and settings for the model. You can modify this file to change the model's behavior. Key parameters include:

## Project Structure

- `models/`: Contains the U-Net and DMG model implementations
- `datasets/`: Includes the TreeCountingDataset class
- `utils/`: Helper functions and classes for training and evaluation
- `train.py`: Main script for training the model
- `test.py`: Script for evaluating the model on the test set
- `visualizer.py`: Script for visualizing density maps

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
