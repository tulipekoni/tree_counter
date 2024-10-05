# Kernel-based Density Map Generation for Dense Object Counting

## Data preparation

## Pretrained model

The pretrained model can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1TaY5I1eHIt7pm2YBfqw4BfnpX2l3Bof4?usp=sharing).

### Prepare

1、 Download Yosemite Tree Dataset [link](https://drive.google.com/drive/folders/1NWAqslICPoTS8OvT8zosI0R7cmsl6x9j)

2、 Pre-Process Data (split large image and split train/validation)

```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```

3、 Train model (validate on single GTX Titan X)

## Test

```
python test.py --data-dir ./data --save-dir ./checkpoints/model.pth --net csrnet --device 0
```

## Train

```

python train.py --net vgg19 --data-dir PATH_TO_DATASET --save-dir PATH_TO_CHECKPOINT

```

### Citation

If you use our code or models in your research, please cite with:

```

```
