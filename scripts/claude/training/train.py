"""
Main training function for wildlife classification with optimized settings
"""

import torch
import multiprocessing as mp
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from ..models.mobilenet import WildlifeMobileNet
from ..datasets import create_train_val_datasets
from ..utils import setup_cuda_multiprocessing
from .trainer import WildlifeTrainer


def train_model(
    data_dir: str,
    yolo_model_path: str,
    num_classes: Optional[int] = None,
    batch_size: int = 16,
    stage1_epochs: int = 30,
    stage2_epochs: int = 5,
    yolo_conf_threshold: float = 0.15,
    fallback_conf_threshold: float = 0.1,
    fallback_strategy: str = "smart_crop",
    train_augmentation: str = "minimal",  # Fix double augmentation
    val_augmentation: str = "none",       # No validation augmentation
    device: Optional[torch.device] = None,
    save_dir: str = "models/claude_checkpoints",
    use_cache: bool = True,
    log_progress: bool = True
) -> str:
    """
    Complete training pipeline for wildlife classification with optimized settings
    
    Args:
        data_dir: Path to dataset directory
        yolo_model_path: Path to YOLO model for preprocessing
        num_classes: Number of classes (auto-detected if None)
        batch_size: Training batch size
        stage1_epochs: Epochs for stage 1 (frozen backbone)
        stage2_epochs: Epochs for stage 2 (fine-tuning)
        yolo_conf_threshold: YOLO confidence threshold (optimized default: 0.15)
        fallback_conf_threshold: Fallback YOLO threshold (default: 0.1)
        fallback_strategy: Fallback strategy when YOLO fails
        train_augmentation: Training augmentation mode ("none", "minimal", "standard")
        val_augmentation: Validation augmentation mode (should be "none")
        device: Training device (auto-detected if None)
        save_dir: Directory to save models and checkpoints
        use_cache: Whether to cache YOLO crops
        log_progress: Whether to log detailed progress
    
    Returns:
        Path to saved final model
    """
    
    if log_progress:
        print("🔥 Starting Optimized Wildlife Training Pipeline")
        print("=" * 60)
    
    # Step 1: Setup multiprocessing for CUDA
    setup_cuda_multiprocessing()
    
    # Step 2: Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if log_progress:
        print(f"Device: {device}")
        print(f"Data: {data_dir}")
        print(f"YOLO: {yolo_model_path}")
        print(f"Augmentation: train={train_augmentation}, val={val_augmentation}")
    
    # Step 3: Validate inputs
    data_path = Path(data_dir)
    yolo_path = Path(yolo_model_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    
    # Step 4: Create datasets with optimized settings
    if log_progress:
        print(f"\n📦 Creating optimized datasets...")
        print(f"  - YOLO threshold: {yolo_conf_threshold}")
        print(f"  - Fallback threshold: {fallback_conf_threshold}")
        print(f"  - Fallback strategy: {fallback_strategy}")
    
    train_dataset, val_dataset = create_train_val_datasets(
        data_dir=data_dir,
        yolo_model_path=yolo_model_path,
        val_split=0.2,
        train_augmentation=train_augmentation,
        val_augmentation=val_augmentation,
        yolo_conf_threshold=yolo_conf_threshold,
        fallback_conf_threshold=fallback_conf_threshold,
        fallback_strategy=fallback_strategy,
        use_cache=use_cache,
        log_failures=log_progress
    )
    
    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = len(train_dataset.class_names)
    
    class_names = train_dataset.class_names
    
    if log_progress:
        print(f"✓ Datasets created: {len(train_dataset)} train, {len(val_dataset)} val")
        print(f"Classes: {class_names}")
    
    # Step 5: Create data loaders with CUDA-safe settings
    # Use num_workers=0 to avoid CUDA multiprocessing issues
    safe_num_workers = 0 if torch.cuda.is_available() else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=safe_num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=safe_num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    # Step 6: Create model and trainer
    model = WildlifeMobileNet(num_classes=num_classes, pretrained=True)
    trainer = WildlifeTrainer(model, device, save_dir=save_dir)
    
    if log_progress:
        print(f"\n🧠 Model initialized: {num_classes} classes")
        model.print_trainable_parameters()
    
    # Step 7: Stage 1 Training (Frozen backbone)
    if log_progress:
        print(f"\n🚀 Stage 1: Frozen backbone training ({stage1_epochs} epochs)")
    
    stage1_history = trainer.train_stage1(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=stage1_epochs,
        lr=0.001,
        patience=5
    )
    
    # Print dataset statistics after Stage 1
    if log_progress:
        print(f"\n📊 Training dataset statistics after Stage 1:")
        train_dataset.print_statistics()
    
    # Step 8: Stage 2 Training (Fine-tuning)
    if stage2_epochs > 0:
        if log_progress:
            print(f"\n🔥 Stage 2: Fine-tuning ({stage2_epochs} epochs)")
        
        stage2_history = trainer.train_stage2(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=stage2_epochs,
            lr=1e-5,
            unfreeze_layers=20,
            patience=3
        )
    else:
        stage2_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Step 9: Save final model
    final_model_path = trainer.save_final_model(class_names)
    
    # Step 10: Generate training plots
    if log_progress:
        trainer.plot_training_history(stage1_history, stage2_history)
    
    # Step 11: Final statistics and summary
    if log_progress:
        # Calculate final metrics
        stage1_best = max(stage1_history['val_acc']) if stage1_history['val_acc'] else 0
        stage2_best = max(stage2_history['val_acc']) if stage2_history['val_acc'] else stage1_best
        improvement = stage2_best - stage1_best
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Final model saved: {final_model_path}")
        print(f"📊 Training Summary:")
        print(f"  - Stage 1 best accuracy: {stage1_best:.2f}%")
        print(f"  - Stage 2 best accuracy: {stage2_best:.2f}%")
        print(f"  - Improvement from fine-tuning: {improvement:+.2f}%")
        
        # Final dataset statistics
        val_stats = val_dataset.get_failure_statistics()
        print(f"\n📊 Final Dataset Performance:")
        print(f"  - YOLO success rate: {val_stats['yolo_success_rate']:.2%}")
        print(f"  - Fallback usage rate: {val_stats['total_fallback_rate']:.2%}")
        
        if 'crop_quality' in val_stats:
            print(f"  - Average crop quality: {val_stats['crop_quality']['mean']:.3f}")
    
    return str(final_model_path)