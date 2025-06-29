"""
Training pipeline for wildlife classification
"""

from .trainer import WildlifeTrainer
from .train import train_model

__all__ = ['WildlifeTrainer', 'train_model']