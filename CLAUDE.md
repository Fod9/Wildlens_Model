# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wildlens_Model is a computer vision project focused on wildlife footprint detection and classification. The project combines YOLO (PyTorch) for footprint detection with MobileNet (TensorFlow) for species classification, creating a two-stage pipeline for wildlife tracking from footprint images.

## Architecture

### Two-Stage TensorFlow MobileNet Training Pipeline
1. **Stage 1: OAT Feature Extraction** - Train MobileNet on OpenAnimalTracks dataset with YOLO preprocessing
2. **Stage 2: Real Dataset Fine-tuning** - Fine-tune the model on real dataset with lower learning rate

### Key Components
- `scripts/yolo_finetuning/yolo_inference.py` - YOLO preprocessing for MobileNet training data
- `notebooks/models/mobilenet.ipynb` - Main TensorFlow MobileNet training notebook
- `MobileNet_training.md` - Detailed training specifications and requirements

### Data Structure
- `data/OpenAnimalTracks_spokay/cropped_imgs/` - OAT dataset with train/test/val splits (18 classes)
- `data/dataset_no_oat_downsample_spokay/` - Real dataset (13 classes, needs train/val split)
- Both datasets use different structures and require `image_dataset_from_directory()` loading

## Development Workflows

### MobileNet Training (TensorFlow)
MobileNet training is performed in `notebooks/models/mobilenet.ipynb` with GPU acceleration:
- Uses CUDA when available (RTX 5070 GPU detected in notebooks)
- Two-stage training: OAT dataset → Real dataset fine-tuning
- Models saved as `.keras` files in `notebooks/models/`

### YOLO Preprocessing for MobileNet
Use `YOLOInference` class to preprocess training data:
```python
yolo_inference = YOLOInference('notebooks/yolo/best_so_far.pt', conf_threshold=0.25, iou_threshold=0.45)
result = yolo_inference.infer_and_get_best_crop(image_path_or_numpy_array)
# Use result for MobileNet training data
```

### Data Processing
- `scripts/yolo_finetuning/yolo_clean_dataset.py` - Dataset cleaning
- `scripts/yolo_finetuning/yolo_train_val_split.py` - Train/validation splitting
- YOLO preprocessing generates crops for MobileNet training

## Key Files and Classes

### Core Inference Classes
- `YOLOInference` (`scripts/yolo_finetuning/yolo_inference.py:98`) - Main inference class
- `BBoxUtility` (`scripts/yolo_finetuning/yolo_inference.py:51`) - Bounding box operations
- `AreaUtility` (`scripts/yolo_finetuning/yolo_inference.py:24`) - Area calculations and scoring
- `BBoxWithScore` (`scripts/yolo_finetuning/yolo_inference.py:16`) - Bounding box data structure

### Model Files
- Best YOLO model: `notebooks/yolo/best_so_far.pt`
- MobileNet models: `notebooks/models/mobilenet.keras`, `notebooks/models/mobilenet_finetuned.keras`
- Base YOLO models: `notebooks/yolo/yolo11n.pt`, `notebooks/yolo/yolov8n.pt`

## Dependencies

The project uses:
- **TensorFlow/Keras** for MobileNet classification training
- **PyTorch** with Ultralytics YOLO for footprint detection preprocessing
- PIL/OpenCV for image processing
- NumPy for numerical operations
- Jupyter notebooks for experimentation

## Training Pipeline

### Two-Stage TensorFlow MobileNet Training
1. **Stage 1: OAT Feature Extraction**
   - Load OAT dataset using `keras.preprocessing.image_dataset_from_directory()`
   - Apply data augmentation (rotations, flips, brightness, zoom, contrast)
   - Train MobileNetV3Small with frozen backbone on 18 classes
   - Save model as `mobilenet_oat.keras`

2. **Stage 2: Real Dataset Fine-tuning**
   - Load real dataset from `data/dataset_no_oat_downsample_spokay/`
   - Create 80/20 train/validation split
   - Unfreeze last N layers of MobileNet backbone
   - Fine-tune with lower learning rate (1e-5) on 13 classes
   - Save final model as `mobilenet_finetuned_with_no_oat.keras`

### Training Usage

```python
# In notebooks/models/mobilenet.ipynb
# Stage 1: Train on OAT dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "../../data/OpenAnimalTracks_spokay/cropped_imgs/train"
)
# ... training code ...

# Stage 2: Fine-tune on real dataset
real_ds = keras.preprocessing.image_dataset_from_directory(
    "../../data/dataset_no_oat_downsample_spokay"
)
# ... fine-tuning code ...
```

## Current Branch

Working on `feature/mobile_net_plus_yolo` branch which implements the combined detection and classification pipeline.