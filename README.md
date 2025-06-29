# Wildlens_Model

Wildlens footprint detection and classification using YOLO + MobileNet pipeline.

## 🎯 Overview

This project combines YOLO (PyTorch) for footprint detection with MobileNet (TensorFlow) for species classification, creating a two-stage pipeline for wildlife tracking from footprint images.

## 🏗️ Architecture

### Two-Stage TensorFlow MobileNet Training
1. **Stage 1: OAT Feature Extraction** - Train MobileNet on OpenAnimalTracks dataset (18 classes)
2. **Stage 2: Real Dataset Fine-tuning** - Fine-tune on real dataset (13 classes) with lower learning rate

### Key Components
- **YOLO Detection**: Uses `scripts/yolo_finetuning/yolo_inference.py` for footprint localization
- **MobileNet Training**: TensorFlow/Keras training in `notebooks/models/mobilenet.ipynb`
- **Data Processing**: YOLO preprocessing for MobileNet training data

## 📁 Data Structure

- `data/OpenAnimalTracks_spokay/cropped_imgs/` - OAT dataset with train/test/val splits (18 classes)
- `data/dataset_no_oat_downsample_spokay/` - Real dataset (13 classes, needs train/val split)

## 🚀 Quick Start

### Training MobileNet
Open and run `notebooks/models/mobilenet.ipynb` for the complete two-stage training pipeline.

### YOLO Preprocessing
```python
from scripts.yolo_finetuning.yolo_inference import YOLOInference

yolo_inference = YOLOInference('notebooks/yolo/best_so_far.pt')
result = yolo_inference.infer_and_get_best_crop(image_path)
```

## 📋 Requirements

- TensorFlow/Keras for MobileNet training
- PyTorch + Ultralytics for YOLO preprocessing
- CUDA-compatible GPU recommended

## 📖 Documentation

- `MobileNet_training.md` - Detailed training specifications
- `CLAUDE.md` - Development guidelines and project structure
