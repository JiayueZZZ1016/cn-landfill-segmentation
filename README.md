# High-resolution mapping of regulated landfills in China: A remote sensing dataset for automated detection

## Description

This repository provides a complete implementation of a remote sensing-based semantic segmentation framework for landfill detection in China. The codebase supports data preparation, model training, evaluation, and visualization of results. It is written in Python and based on the PyTorch deep learning framework.

## Repository Structure

- `Dataset.py` — Data loader for remote sensing images and label masks.
- `Train.py` — Training script for segmentation models.
- `Val.py` — Validation script used during training to monitor performance.
- `Test.py` — Script for evaluating model performance on test data.
- `Losses.py` — Implementation of custom loss functions (e.g., Focal Loss, Dice Loss).
- `Metrics.py` — Evaluation metrics including IoU, F1 Score, etc.
- `Optimizers.py` — Optimizer configurations for training.
- `Schedulers.py` — Learning rate scheduler definitions.
- `Crop.py` — Image patch extraction for memory-efficient processing.
- `Pattle.py` — Patch-wise inference utility.
- `Splits.py` — Dataset splitting utility.
- `VisPreds.py` — Visualization script for predicted masks.
- `resnet.py` — Backbone ResNet encoder implementation.
- `unet_resnet.py` — U-Net model with ResNet backbone for semantic segmentation.

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- numpy, pandas, matplotlib, seaborn
- rasterio, GDAL (for geospatial data)
- scikit-learn

Install dependencies via:
```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/JiayueZZZ1016/cn-landfill-segmentation.git
   cd cn-landfill-segmentation
   ```

2. Prepare dataset folders:
   ```
   ├── images/      # GeoTIFF imagery
   ├── labels/      # Corresponding binary label masks
   └── attributes.csv  # Optional vector attributes
   ```

3. Train a model:
   ```bash
   python Train.py
   ```

4. Evaluate the trained model:
   ```bash
   python Test.py
   ```

## License

This project is released under the MIT License.
