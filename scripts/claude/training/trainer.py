"""
Wildlife training pipeline with two-stage training and comprehensive monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
import copy
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..models.mobilenet import WildlifeMobileNet


class WildlifeTrainer:
    """
    Training pipeline for wildlife footprint classification using YOLO preprocessing
    """
    
    def __init__(
        self,
        model: WildlifeMobileNet,
        device: torch.device,
        save_dir: str = "models/pytorch_checkpoints"
    ):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model.to(device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        print(f"✓ WildlifeTrainer initialized on {device}")
    
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer, 
        criterion: nn.Module,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(
        self, 
        val_loader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/(len(pbar.iterable) if hasattr(pbar, "iterable") else 1):.3f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_stage1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 5
    ) -> Dict:
        """
        Stage 1: Train classifier head with frozen backbone
        (Equivalent to initial training in TensorFlow version)
        """
        print(f"\n🚀 Starting Stage 1: Frozen backbone training ({epochs} epochs)")
        
        # Freeze backbone
        self.model.freeze_backbone()
        self.model.print_trainable_parameters()
        
        # Setup optimizer and criterion
        optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=lr,
            weight_decay=0.01  # L2 regularization like TF version
        )
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        stage1_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update history
            stage1_history['train_loss'].append(train_loss)
            stage1_history['train_acc'].append(train_acc)
            stage1_history['val_loss'].append(val_loss)
            stage1_history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                # Save checkpoint
                self.save_checkpoint(epoch, train_loss, val_acc, "stage1_best")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"✅ Stage 1 completed. Best validation accuracy: {best_val_acc:.2f}%")
        return stage1_history
    
    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-5,
        unfreeze_layers: int = 20,
        patience: int = 3
    ) -> Dict:
        """
        Stage 2: Fine-tune last layers of backbone
        (Equivalent to fine-tuning in TensorFlow version)
        """
        print(f"\n🔥 Starting Stage 2: Fine-tuning last {unfreeze_layers} layers ({epochs} epochs)")
        
        # Unfreeze last layers
        self.model.unfreeze_backbone(unfreeze_last_n_layers=unfreeze_layers)
        self.model.print_trainable_parameters()
        
        # Setup optimizer with lower learning rate
        optimizer = optim.Adam(
            self.model.get_trainable_parameters(),
            lr=lr  # Much lower LR for fine-tuning
        )
        criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        
        # Early stopping
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        stage2_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            stage2_history['train_loss'].append(train_loss)
            stage2_history['train_acc'].append(train_acc)
            stage2_history['val_loss'].append(val_loss)
            stage2_history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")
            
            # Early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                # Save checkpoint
                self.save_checkpoint(epoch, train_loss, val_acc, "stage2_best")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"✅ Stage 2 completed. Best validation accuracy: {best_val_acc:.2f}%")
        return stage2_history
    
    def save_checkpoint(self, epoch: int, loss: float, accuracy: float, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'model_config': {
                'num_classes': len(self.model.classifier[-1].weight),
            }
        }
        
        checkpoint_path = self.save_dir / f"{name}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self, class_names: List[str]):
        """Save final trained model with metadata"""
        model_info = {
            'model_state_dict': self.model.state_dict(),
            'class_names': class_names,
            'model_architecture': 'WildlifeMobileNet',
            'input_size': (224, 224),
            'num_classes': len(class_names),
            'training_completed': True
        }
        
        final_path = self.save_dir / "wildlife_mobilenet_final.pth"
        torch.save(model_info, final_path)
        print(f"🎯 Final model saved: {final_path}")
        return final_path
    
    def plot_training_history(self, stage1_history: Dict, stage2_history: Dict):
        """Plot training curves for both stages"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Stage 1 Loss
        ax1.plot(stage1_history['train_loss'], label='Train')
        ax1.plot(stage1_history['val_loss'], label='Validation')
        ax1.set_title('Stage 1: Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Stage 1 Accuracy
        ax2.plot(stage1_history['train_acc'], label='Train')
        ax2.plot(stage1_history['val_acc'], label='Validation')
        ax2.set_title('Stage 1: Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Stage 2 Loss
        if stage2_history['train_loss']:
            ax3.plot(stage2_history['train_loss'], label='Train')
            ax3.plot(stage2_history['val_loss'], label='Validation')
            ax3.set_title('Stage 2: Loss')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True)
        
        # Stage 2 Accuracy
        if stage2_history['train_acc']:
            ax4.plot(stage2_history['train_acc'], label='Train')
            ax4.plot(stage2_history['val_acc'], label='Validation')
            ax4.set_title('Stage 2: Accuracy')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy (%)')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.show()