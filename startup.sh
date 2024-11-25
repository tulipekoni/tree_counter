#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting repository setup..."

# Check if Python is installed
if ! command_exists python3; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 before continuing.${NC}"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating required directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p processed_data

# Download dataset
echo "Downloading dataset..."
gdown --folder https://drive.google.com/drive/folders/1NWAqslICPoTS8OvT8zosI0R7cmsl6x9j -O ./data

# Preprocess dataset
echo "Preprocessing dataset..."
python ./utils/preprocess_dataset.py

echo -e "${GREEN}Setup complete! You can now:${NC}"
echo "1. Train the model: python train.py"
echo "2. Test the model: python test.py --model_folder ./path_to_model_folder"
echo "3. Visualize results: python visualizer.py --model_dir ./path_to_model_folder" 