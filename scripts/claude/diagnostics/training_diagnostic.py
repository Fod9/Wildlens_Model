"""
Training diagnostic tools to analyze MobileNet training performance and data issues
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from ..models.mobilenet import WildlifeMobileNet


class TrainingDiagnostic:
    """
    Comprehensive diagnostic tool for training performance analysis
    """
    
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        """
        Args:
            model_path: Path to trained model
            device: Device for inference (auto-detected if None)
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"🔍 Loading model: {model_path}")
        self.model, self.class_names = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Training diagnostic initialized")
        print(f"  - Model: {model_path}")
        print(f"  - Classes: {len(self.class_names)} ({self.class_names})")
        print(f"  - Device: {self.device}")
    
    def _load_model(self) -> Tuple[WildlifeMobileNet, List[str]]:
        """Load trained model with metadata"""
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Extract metadata
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            num_classes = len(class_names)
        elif 'model_config' in checkpoint and 'num_classes' in checkpoint['model_config']:
            num_classes = checkpoint['model_config']['num_classes']
            class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            raise ValueError("Could not determine number of classes from model checkpoint")
        
        # Create model
        model = WildlifeMobileNet(num_classes=num_classes, pretrained=False)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model, class_names
    
    def analyze_dataloader(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> Dict:
        """
        Analyze model performance on a dataloader
        
        Args:
            dataloader: DataLoader to analyze
            max_batches: Maximum number of batches to analyze (None for all)
            
        Returns:
            Comprehensive analysis results
        """
        
        print(f"🔍 Analyzing model performance on dataloader")
        
        self.model.eval()
        
        results = {
            'predictions': [],
            'true_labels': [],
            'confidences': [],
            'logits': [],
            'batch_stats': [],
            'misclassifications': [],
            'class_performance': defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
        }
        
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else float('inf')
        if max_batches:
            total_batches = min(total_batches, max_batches)
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                if batch_idx % 10 == 0:
                    print(f"Progress: {batch_idx}/{total_batches}")
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                logits = self.model(data)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # Store results
                results['predictions'].extend(predictions.cpu().numpy())
                results['true_labels'].extend(target.cpu().numpy())
                results['confidences'].extend(confidences.cpu().numpy())
                results['logits'].extend(logits.cpu().numpy())
                
                # Batch statistics
                batch_acc = (predictions == target).float().mean().item()
                batch_conf = confidences.mean().item()
                
                results['batch_stats'].append({
                    'batch_idx': batch_idx,
                    'accuracy': batch_acc,
                    'avg_confidence': batch_conf,
                    'size': len(target)
                })
                
                # Per-sample analysis
                for i in range(len(target)):
                    true_class = target[i].item()
                    pred_class = predictions[i].item()
                    confidence = confidences[i].item()
                    
                    true_class_name = self.class_names[true_class]
                    
                    # Update class performance
                    results['class_performance'][true_class_name]['total'] += 1
                    results['class_performance'][true_class_name]['confidences'].append(confidence)
                    
                    if true_class == pred_class:
                        results['class_performance'][true_class_name]['correct'] += 1
                    else:
                        # Store misclassification
                        results['misclassifications'].append({
                            'true_class': true_class_name,
                            'pred_class': self.class_names[pred_class],
                            'confidence': confidence,
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
        
        # Calculate summary statistics
        results['summary'] = self._calculate_training_summary(results)
        
        return results
    
    def _calculate_training_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics from training analysis"""
        
        predictions = np.array(results['predictions'])
        true_labels = np.array(results['true_labels'])
        confidences = np.array(results['confidences'])
        
        overall_accuracy = np.mean(predictions == true_labels)
        avg_confidence = np.mean(confidences)
        
        summary = {
            'overall_accuracy': overall_accuracy,
            'avg_confidence': avg_confidence,
            'total_samples': len(predictions),
            'num_misclassifications': len(results['misclassifications']),
            'confidence_std': np.std(confidences)
        }
        
        # Per-class accuracy
        summary['class_accuracies'] = {}
        for class_name, stats in results['class_performance'].items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                avg_conf = np.mean(stats['confidences'])
                summary['class_accuracies'][class_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_conf,
                    'total_samples': stats['total']
                }
        
        return summary
    
    def print_training_report(self, results: Dict):
        """Print comprehensive training analysis report"""
        
        summary = results['summary']
        
        print("\n" + "="*60)
        print("🧠 MOBILENET TRAINING ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 Overall Performance:")
        print(f"  - Overall accuracy: {summary['overall_accuracy']:.2%}")
        print(f"  - Average confidence: {summary['avg_confidence']:.3f} ± {summary['confidence_std']:.3f}")
        print(f"  - Total samples: {summary['total_samples']}")
        print(f"  - Misclassifications: {summary['num_misclassifications']} ({summary['num_misclassifications']/summary['total_samples']:.2%})")
        
        print(f"\n📈 Per-Class Performance:")
        class_accs = summary['class_accuracies']
        sorted_classes = sorted(class_accs.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for class_name, stats in sorted_classes:
            print(f"  - {class_name}: {stats['accuracy']:.2%} "
                  f"(conf: {stats['avg_confidence']:.3f}, n: {stats['total_samples']})")
        
        # Identify problematic classes
        worst_classes = sorted_classes[-3:]
        print(f"\n⚠️  Classes with lowest accuracy:")
        for class_name, stats in worst_classes:
            print(f"  - {class_name}: {stats['accuracy']:.2%}")
        
        # Confidence analysis
        confidences = np.array(results['confidences'])
        low_conf_count = np.sum(confidences < 0.5)
        medium_conf_count = np.sum((confidences >= 0.5) & (confidences < 0.8))
        high_conf_count = np.sum(confidences >= 0.8)
        
        print(f"\n📊 Confidence Distribution:")
        print(f"  - Low confidence (<0.5): {low_conf_count} ({low_conf_count/len(confidences):.2%})")
        print(f"  - Medium confidence (0.5-0.8): {medium_conf_count} ({medium_conf_count/len(confidences):.2%})")
        print(f"  - High confidence (>0.8): {high_conf_count} ({high_conf_count/len(confidences):.2%})")
        
        # Common misclassifications
        if results['misclassifications']:
            misclass_pairs = [(m['true_class'], m['pred_class']) for m in results['misclassifications']]
            common_mistakes = Counter(misclass_pairs).most_common(5)
            
            print(f"\n🔍 Most Common Misclassifications:")
            for (true_class, pred_class), count in common_mistakes:
                print(f"  - {true_class} → {pred_class}: {count} times")
    
    def plot_training_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Create visualization plots for training analysis"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Confidence distribution
        confidences = results['confidences']
        ax1.hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Prediction Confidence')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Per-class accuracy
        class_names = list(results['summary']['class_accuracies'].keys())
        accuracies = [results['summary']['class_accuracies'][name]['accuracy'] 
                     for name in class_names]
        
        bars = ax2.bar(range(len(class_names)), accuracies, 
                      color=['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' 
                             for acc in accuracies])
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Per-Class Accuracy')
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=8)
        
        # 3. Confusion matrix
        if len(results['predictions']) > 0:
            cm = confusion_matrix(results['true_labels'], results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                       xticklabels=class_names, yticklabels=class_names)
            ax3.set_xlabel('Predicted Class')
            ax3.set_ylabel('True Class')
            ax3.set_title('Confusion Matrix')
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='y', rotation=0)
        
        # 4. Batch accuracy over time
        batch_stats = results['batch_stats']
        batch_accuracies = [b['accuracy'] for b in batch_stats]
        batch_indices = [b['batch_idx'] for b in batch_stats]
        
        ax4.plot(batch_indices, batch_accuracies, alpha=0.7, color='green')
        ax4.set_xlabel('Batch Index')
        ax4.set_ylabel('Batch Accuracy')
        ax4.set_title('Accuracy Over Batches')
        ax4.grid(True, alpha=0.3)
        
        # Add moving average
        if len(batch_accuracies) > 10:
            window_size = min(10, len(batch_accuracies) // 4)
            moving_avg = np.convolve(batch_accuracies, np.ones(window_size)/window_size, mode='valid')
            moving_indices = batch_indices[window_size-1:]
            ax4.plot(moving_indices, moving_avg, color='red', linewidth=2, 
                    label=f'Moving Average (window={window_size})')
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training analysis plots saved: {save_path}")
        
        plt.show()
    
    def analyze_model_weights(self) -> Dict:
        """Analyze model weights for potential issues"""
        
        print("🔍 Analyzing model weights...")
        
        weight_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy()
                
                weight_stats[name] = {
                    'shape': weights.shape,
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'min': float(np.min(weights)),
                    'max': float(np.max(weights)),
                    'zeros_ratio': float(np.mean(weights == 0)),
                    'small_weights_ratio': float(np.mean(np.abs(weights) < 1e-6))
                }
        
        return weight_stats
    
    def test_data_augmentation_impact(self, dataset, num_samples: int = 100) -> Dict:
        """
        Test impact of different augmentation modes on the same samples
        
        Args:
            dataset: Dataset with configurable augmentation
            num_samples: Number of samples to test
            
        Returns:
            Analysis of augmentation impact
        """
        
        print(f"🔍 Testing data augmentation impact on {num_samples} samples")
        
        if not hasattr(dataset, 'set_augmentation_mode'):
            print("Dataset does not support configurable augmentation")
            return {}
        
        augmentation_modes = ['none', 'minimal', 'standard']
        results = {mode: {'predictions': [], 'confidences': []} for mode in augmentation_modes}
        
        # Sample indices to test
        test_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        for mode in augmentation_modes:
            print(f"Testing augmentation mode: {mode}")
            dataset.set_augmentation_mode(mode)
            
            for idx in test_indices:
                try:
                    data, label = dataset[idx]
                    data = data.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        logits = self.model(data)
                        probabilities = F.softmax(logits, dim=1)
                        prediction = torch.argmax(probabilities, dim=1).item()
                        confidence = torch.max(probabilities, dim=1)[0].item()
                    
                    results[mode]['predictions'].append(prediction)
                    results[mode]['confidences'].append(confidence)
                    
                except Exception as e:
                    print(f"Error processing sample {idx} with mode {mode}: {e}")
        
        # Analyze consistency across augmentation modes
        analysis = {
            'mode_comparison': {},
            'consistency_score': 0,
            'confidence_differences': {}
        }
        
        if len(augmentation_modes) > 1:
            # Compare predictions between modes
            base_mode = augmentation_modes[0]
            base_preds = results[base_mode]['predictions']
            
            for mode in augmentation_modes[1:]:
                mode_preds = results[mode]['predictions']
                consistency = np.mean(np.array(base_preds) == np.array(mode_preds))
                analysis['mode_comparison'][f"{base_mode}_vs_{mode}"] = consistency
            
            # Average consistency
            analysis['consistency_score'] = np.mean(list(analysis['mode_comparison'].values()))
            
            # Confidence differences
            for mode in augmentation_modes:
                mode_conf = np.array(results[mode]['confidences'])
                analysis['confidence_differences'][mode] = {
                    'mean': float(np.mean(mode_conf)),
                    'std': float(np.std(mode_conf))
                }
        
        analysis['raw_results'] = results
        
        return analysis