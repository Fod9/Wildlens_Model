"""
YOLO model diagnostic tools to analyze detection performance
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import sys

# Import YOLO inference
try:
    from scripts.yolo_finetuning.yolo_inference import YOLOInference
except ImportError as e:
    print(f"Warning: Could not import YOLOInference: {e}")


class YOLODiagnostic:
    """
    Comprehensive diagnostic tool for YOLO model performance analysis
    """
    
    def __init__(self, yolo_model_path: str, conf_threshold: float = 0.15):
        """
        Args:
            yolo_model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.yolo_model_path = yolo_model_path
        self.conf_threshold = conf_threshold
        
        # Initialize YOLO
        print(f"🔍 Loading YOLO model: {yolo_model_path}")
        self.yolo = YOLOInference(yolo_model_path, conf_threshold=conf_threshold)
        
        print(f"✓ YOLO Diagnostic initialized (threshold: {conf_threshold})")
    
    def analyze_dataset(self, dataset_dir: str, max_images: Optional[int] = None) -> Dict:
        """
        Analyze YOLO performance across entire dataset
        
        Args:
            dataset_dir: Path to dataset directory
            max_images: Maximum number of images to analyze (None for all)
            
        Returns:
            Comprehensive analysis results
        """
        
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        print(f"🔍 Analyzing YOLO performance on dataset: {dataset_path}")
        
        # Collect all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = []
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_images = [
                    img for img in class_dir.iterdir() 
                    if img.suffix.lower() in image_extensions
                ]
                all_images.extend([(img, class_dir.name) for img in class_images])
        
        if max_images:
            all_images = all_images[:max_images]
        
        print(f"Found {len(all_images)} images to analyze")
        
        # Analysis results
        results = {
            'total_images': len(all_images),
            'detections': [],
            'failures': [],
            'class_stats': defaultdict(lambda: {'total': 0, 'detected': 0, 'failed': 0}),
            'confidence_distribution': [],
            'bbox_sizes': [],
            'processing_times': []
        }
        
        # Process each image
        for i, (image_path, class_name) in enumerate(all_images):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(all_images)} ({i/len(all_images)*100:.1f}%)")
            
            try:
                # Run YOLO inference
                import time
                start_time = time.time()
                
                yolo_result = self.yolo.infer_and_get_best_crop(str(image_path))
                
                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)
                
                # Update class stats
                results['class_stats'][class_name]['total'] += 1
                
                if yolo_result is not None:
                    best_bbox, cropped_image = yolo_result
                    
                    results['detections'].append({
                        'image_path': str(image_path),
                        'class': class_name,
                        'confidence': best_bbox.confidence,
                        'bbox': best_bbox.bbox,
                        'area': best_bbox.area,
                        'crop_shape': cropped_image.shape,
                        'processing_time': processing_time
                    })
                    
                    results['class_stats'][class_name]['detected'] += 1
                    results['confidence_distribution'].append(best_bbox.confidence)
                    results['bbox_sizes'].append(best_bbox.area)
                    
                else:
                    results['failures'].append({
                        'image_path': str(image_path),
                        'class': class_name,
                        'processing_time': processing_time
                    })
                    
                    results['class_stats'][class_name]['failed'] += 1
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results['failures'].append({
                    'image_path': str(image_path),
                    'class': class_name,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_stats(results)
        
        return results
    
    def _calculate_summary_stats(self, results: Dict) -> Dict:
        """Calculate summary statistics from analysis results"""
        
        total_images = results['total_images']
        successful_detections = len(results['detections'])
        failed_detections = len(results['failures'])
        
        summary = {
            'detection_rate': successful_detections / total_images if total_images > 0 else 0,
            'failure_rate': failed_detections / total_images if total_images > 0 else 0,
            'avg_confidence': np.mean(results['confidence_distribution']) if results['confidence_distribution'] else 0,
            'std_confidence': np.std(results['confidence_distribution']) if results['confidence_distribution'] else 0,
            'avg_bbox_area': np.mean(results['bbox_sizes']) if results['bbox_sizes'] else 0,
            'avg_processing_time': np.mean(results['processing_times']) if results['processing_times'] else 0
        }
        
        # Per-class statistics
        summary['class_performance'] = {}
        for class_name, stats in results['class_stats'].items():
            if stats['total'] > 0:
                summary['class_performance'][class_name] = {
                    'detection_rate': stats['detected'] / stats['total'],
                    'total_images': stats['total'],
                    'successful': stats['detected'],
                    'failed': stats['failed']
                }
        
        return summary
    
    def print_analysis_report(self, results: Dict):
        """Print comprehensive analysis report"""
        
        summary = results['summary']
        
        print("\n" + "="*60)
        print("🔍 YOLO PERFORMANCE ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 Overall Statistics:")
        print(f"  - Total images analyzed: {results['total_images']}")
        print(f"  - Successful detections: {len(results['detections'])} ({summary['detection_rate']:.2%})")
        print(f"  - Failed detections: {len(results['failures'])} ({summary['failure_rate']:.2%})")
        print(f"  - Average confidence: {summary['avg_confidence']:.3f} ± {summary['std_confidence']:.3f}")
        print(f"  - Average bbox area: {summary['avg_bbox_area']:.0f} pixels²")
        print(f"  - Average processing time: {summary['avg_processing_time']:.3f}s")
        
        print(f"\n📈 Per-Class Performance:")
        for class_name, perf in summary['class_performance'].items():
            print(f"  - {class_name}: {perf['successful']}/{perf['total_images']} "
                  f"({perf['detection_rate']:.2%})")
        
        # Identify problematic classes
        worst_classes = sorted(
            summary['class_performance'].items(),
            key=lambda x: x[1]['detection_rate']
        )[:3]
        
        print(f"\n⚠️  Classes with lowest detection rates:")
        for class_name, perf in worst_classes:
            print(f"  - {class_name}: {perf['detection_rate']:.2%}")
        
        # Confidence distribution analysis
        if results['confidence_distribution']:
            conf_array = np.array(results['confidence_distribution'])
            low_conf_count = np.sum(conf_array < 0.3)
            medium_conf_count = np.sum((conf_array >= 0.3) & (conf_array < 0.7))
            high_conf_count = np.sum(conf_array >= 0.7)
            
            print(f"\n📊 Confidence Distribution:")
            print(f"  - Low confidence (<0.3): {low_conf_count} ({low_conf_count/len(conf_array):.2%})")
            print(f"  - Medium confidence (0.3-0.7): {medium_conf_count} ({medium_conf_count/len(conf_array):.2%})")
            print(f"  - High confidence (>0.7): {high_conf_count} ({high_conf_count/len(conf_array):.2%})")
    
    def plot_analysis_results(self, results: Dict, save_path: Optional[str] = None):
        """Create visualization plots for analysis results"""
        
        if not results['detections']:
            print("No detections to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Confidence distribution
        confidences = results['confidence_distribution']
        ax1.hist(confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(self.conf_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.conf_threshold}')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Detection Confidence Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bounding box area distribution
        bbox_areas = results['bbox_sizes']
        ax2.hist(bbox_areas, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Bounding Box Area (pixels²)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Detection Size Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Per-class detection rates
        class_names = list(results['summary']['class_performance'].keys())
        detection_rates = [results['summary']['class_performance'][name]['detection_rate'] 
                          for name in class_names]
        
        bars = ax3.bar(range(len(class_names)), detection_rates, 
                      color=['green' if rate > 0.8 else 'orange' if rate > 0.5 else 'red' 
                             for rate in detection_rates])
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Detection Rate')
        ax3.set_title('Per-Class Detection Rates')
        ax3.set_xticks(range(len(class_names)))
        ax3.set_xticklabels(class_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, detection_rates)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom', fontsize=8)
        
        # 4. Processing time distribution
        processing_times = results['processing_times']
        ax4.hist(processing_times, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.set_xlabel('Processing Time (seconds)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('YOLO Processing Time Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Analysis plots saved: {save_path}")
        
        plt.show()
    
    def test_threshold_sensitivity(self, test_images: List[str], 
                                  thresholds: List[float] = None) -> Dict:
        """
        Test how detection performance varies with confidence threshold
        
        Args:
            test_images: List of image paths to test
            thresholds: List of confidence thresholds to test
            
        Returns:
            Dictionary with threshold sensitivity results
        """
        
        if thresholds is None:
            thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        print(f"🔍 Testing threshold sensitivity on {len(test_images)} images")
        
        results = {
            'thresholds': thresholds,
            'detection_rates': [],
            'avg_confidences': [],
            'detection_counts': []
        }
        
        for threshold in thresholds:
            print(f"Testing threshold: {threshold}")
            
            # Create temporary YOLO instance with new threshold
            temp_yolo = YOLOInference(self.yolo_model_path, conf_threshold=threshold)
            
            detections = 0
            confidences = []
            
            for image_path in test_images:
                try:
                    result = temp_yolo.infer_and_get_best_crop(image_path)
                    if result is not None:
                        detections += 1
                        best_bbox, _ = result
                        confidences.append(best_bbox.confidence)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            
            detection_rate = detections / len(test_images)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            results['detection_rates'].append(detection_rate)
            results['avg_confidences'].append(avg_confidence)
            results['detection_counts'].append(detections)
            
            print(f"  - Detection rate: {detection_rate:.2%}")
            print(f"  - Average confidence: {avg_confidence:.3f}")
        
        return results
    
    def save_failure_examples(self, results: Dict, output_dir: str, max_examples: int = 10):
        """Save examples of failed detections for analysis"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        failures = results['failures']
        if not failures:
            print("No failures to save")
            return
        
        print(f"💾 Saving {min(max_examples, len(failures))} failure examples to {output_path}")
        
        for i, failure in enumerate(failures[:max_examples]):
            try:
                image_path = failure['image_path']
                class_name = failure['class']
                
                # Load and save original image
                image = cv2.imread(image_path)
                if image is not None:
                    output_filename = f"failure_{i+1:02d}_{class_name}_{Path(image_path).name}"
                    cv2.imwrite(str(output_path / output_filename), image)
                    
            except Exception as e:
                print(f"Error saving failure example {i}: {e}")
        
        print(f"✓ Saved failure examples to {output_path}")