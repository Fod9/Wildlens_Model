"""
Claude Wildlife Classification Package

A unified PyTorch pipeline for wildlife footprint detection and classification.
Combines YOLO detection with MobileNet classification in a clean, organized package.

Main Components:
- models: Neural network architectures (MobileNet)
- datasets: YOLO-preprocessed datasets with robust fallbacks
- training: Two-stage training pipeline
- inference: End-to-end inference pipeline
- diagnostics: YOLO and dataset analysis tools
- utils: Utilities and helpers

Example Usage:
    from scripts.claude.training import train_model
    from scripts.claude.inference import WildlifePipeline
    
    # Train model
    model_path = train_model(
        data_dir="data/dataset",
        yolo_model="notebooks/yolo/best_so_far.pt"
    )
    
    # Run inference
    pipeline = WildlifePipeline(yolo_model, model_path)
    result = pipeline.predict("image.jpg")
"""

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"

# Main exports for easy access
from .models.mobilenet import WildlifeMobileNet
from .datasets.yolo_dataset import YOLODataset
from .training.trainer import WildlifeTrainer
from .inference.pipeline import WildlifePipeline

__all__ = [
    'WildlifeMobileNet',
    'YOLODataset', 
    'WildlifeTrainer',
    'WildlifePipeline'
]