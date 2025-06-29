"""
Unified YOLO-preprocessed dataset with robust fallbacks and double augmentation fix
Handles YOLO detection failures gracefully while avoiding double augmentation issues
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import numpy as np
import cv2
import logging
from collections import Counter
from sklearn.model_selection import train_test_split

# Import YOLO inference

try:
    from scripts.yolo_finetuning.yolo_inference import YOLOInference
except ImportError:
    print("Warning: Could not import YOLOInference. Make sure yolo_inference.py is available.")
    YOLOInference = None


class YOLODataset(Dataset):
    """
    Unified YOLO-preprocessed dataset with intelligent fallback strategies
    
    Key Features:
    - Robust YOLO detection with multiple fallback strategies
    - Fixes double augmentation issue (YOLO crops + MobileNet augmentation)
    - Comprehensive failure tracking and statistics
    - Configurable augmentation strategy to avoid over-augmentation
    
    Fallback hierarchy when YOLO fails:
    1. Lower confidence threshold retry
    2. Center crop with edge detection
    3. Smart crop using image analysis
    4. Simple resize of full image
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        yolo_model_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        yolo_conf_threshold: float = 0.15,  # Optimized default
        yolo_iou_threshold: float = 0.45,
        fallback_conf_threshold: float = 0.1,  # Lower threshold for retry
        min_crop_area: int = 1000,
        target_size: Tuple[int, int] = (224, 224),
        use_cache: bool = True,
        fallback_strategy: str = "smart_crop",  # "center_crop", "smart_crop", "edge_detection"
        augmentation_mode: str = "minimal",  # "none", "minimal", "standard" - fixes double augmentation
        log_failures: bool = True,
        crop_quality_threshold: float = 0.3  # Minimum crop quality score
    ):
        """
        Args:
            augmentation_mode: Controls augmentation to avoid double augmentation
                - "none": No augmentation (for when YOLO crops provide sufficient diversity)
                - "minimal": Basic geometric transforms only
                - "standard": Full augmentation suite (use carefully)
            crop_quality_threshold: Minimum quality score for accepting crops
        """
        self.data_dir = Path(data_dir)
        self.yolo_model_path = yolo_model_path
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_iou_threshold = yolo_iou_threshold
        self.fallback_conf_threshold = fallback_conf_threshold
        self.min_crop_area = min_crop_area
        self.target_size = target_size
        self.use_cache = use_cache
        self.fallback_strategy = fallback_strategy
        self.augmentation_mode = augmentation_mode
        self.log_failures = log_failures
        self.crop_quality_threshold = crop_quality_threshold
        
        # Initialize YOLO models (primary and fallback)
        if YOLOInference is None:
            raise ImportError("YOLOInference not available. Check yolo_inference.py import.")
        
        self.yolo_primary = YOLOInference(
            yolo_model_path, 
            conf_threshold=yolo_conf_threshold,
            iou_threshold=yolo_iou_threshold
        )
        
        self.yolo_fallback = YOLOInference(
            yolo_model_path, 
            conf_threshold=fallback_conf_threshold,
            iou_threshold=yolo_iou_threshold
        )
        
        # Discover classes and build file list
        self.class_names = class_names or self._discover_classes()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.samples = self._make_dataset()
        
        # Setup caching
        if self.use_cache:
            self.cache_dir = self.data_dir.parent / f"claude_yolo_cache_{self.data_dir.name}"
            self.cache_dir.mkdir(exist_ok=True)
        
        # Failure tracking
        self.failure_stats = Counter()
        self.fallback_stats = Counter()
        self.crop_quality_stats = []
        
        # Setup transforms based on augmentation mode
        if transform is None:
            self.transform = self._create_transforms()
        else:
            self.transform = transform
            
        print(f"✓ YOLODataset initialized:")
        print(f"  - Primary threshold: {yolo_conf_threshold}")
        print(f"  - Fallback threshold: {fallback_conf_threshold}")
        print(f"  - Fallback strategy: {fallback_strategy}")
        print(f"  - Augmentation mode: {augmentation_mode}")
        print(f"  - Classes: {len(self.class_names)}")
        print(f"  - Samples: {len(self.samples)}")
    
    def _create_transforms(self) -> transforms.Compose:
        """Create transforms based on augmentation mode to avoid double augmentation"""
        
        base_transforms = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if self.augmentation_mode == "none":
            # No augmentation - YOLO crops provide diversity
            return transforms.Compose(base_transforms)
            
        elif self.augmentation_mode == "minimal":
            # Minimal augmentation - only basic geometric transforms
            augment_transforms = [
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return transforms.Compose(augment_transforms)
            
        elif self.augmentation_mode == "standard":
            # Standard augmentation - use carefully to avoid over-augmentation
            augment_transforms = [
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # Reduced from 5.7
                transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced intensity
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
            return transforms.Compose(augment_transforms)
        
        else:
            raise ValueError(f"Unknown augmentation_mode: {self.augmentation_mode}")
    
    def _discover_classes(self) -> List[str]:
        """Discover class names from subdirectories"""
        classes = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                classes.append(item.name)
        return sorted(classes)
    
    def _make_dataset(self) -> List[Tuple[Path, int]]:
        """Build list of (image_path, class_index) tuples"""
        samples = []
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in extensions:
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _get_cache_path(self, image_path: Path, method: str) -> Path:
        """Get cache path for processed image"""
        if not self.use_cache:
            return None
        
        cache_name = f"{image_path.stem}_{method}_{self.yolo_conf_threshold}_{self.fallback_conf_threshold}.jpg"
        return self.cache_dir / image_path.parent.name / cache_name
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Image.Image]:
        """Load processed image from cache"""
        if cache_path and cache_path.exists():
            try:
                return Image.open(cache_path).convert('RGB')
            except Exception as e:
                if self.log_failures:
                    logging.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, image: Image.Image, cache_path: Path):
        """Save processed image to cache"""
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(cache_path, 'JPEG', quality=95)
            except Exception as e:
                if self.log_failures:
                    logging.warning(f"Failed to save cache {cache_path}: {e}")
    
    def _calculate_crop_quality(self, crop: np.ndarray, bbox_info) -> float:
        """Calculate quality score for a crop (0-1, higher is better)"""
        
        if crop is None or crop.size == 0:
            return 0.0
        
        try:
            # Factor 1: Size (larger crops generally better)
            area_score = min(1.0, bbox_info.area / 10000)  # Normalize by reasonable area
            
            # Factor 2: Confidence (YOLO's confidence in detection)
            conf_score = bbox_info.confidence
            
            # Factor 3: Aspect ratio (avoid extremely distorted crops)
            height, width = crop.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # Penalty for non-square crops
            aspect_score = max(0.0, min(1.0, aspect_score))
            
            # Factor 4: Image sharpness (basic edge detection)
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
            edges = cv2.Canny(gray, 50, 150)
            sharpness_score = np.sum(edges) / (crop.shape[0] * crop.shape[1]) / 255.0
            sharpness_score = min(1.0, sharpness_score * 10)  # Normalize
            
            # Weighted combination
            quality_score = (
                0.3 * area_score +
                0.4 * conf_score +
                0.2 * aspect_score +
                0.1 * sharpness_score
            )
            
            return quality_score
            
        except Exception as e:
            if self.log_failures:
                logging.warning(f"Error calculating crop quality: {e}")
            return 0.0
    
    def _yolo_crop_with_fallback(self, image_path: Path) -> Tuple[Optional[Image.Image], str, float]:
        """
        Try YOLO detection with fallback strategies
        
        Returns:
            Tuple of (processed_image, method_used, quality_score)
        """
        
        # Try cache first
        cache_path = self._get_cache_path(image_path, "yolo_primary")
        cached_image = self._load_from_cache(cache_path)
        if cached_image is not None:
            return cached_image, "cache_primary", 1.0  # Assume cached crops are good quality
        
        # Step 1: Primary YOLO detection
        try:
            result = self.yolo_primary.infer_and_get_best_crop(str(image_path))
            
            if result is not None:
                best_bbox, cropped_array = result
                
                # Calculate crop quality
                quality_score = self._calculate_crop_quality(cropped_array, best_bbox)
                
                if best_bbox.area >= self.min_crop_area and quality_score >= self.crop_quality_threshold:
                    cropped_image = self._array_to_pil(cropped_array)
                    self._save_to_cache(cropped_image, cache_path)
                    self.crop_quality_stats.append(quality_score)
                    return cropped_image, "yolo_primary", quality_score
                else:
                    if best_bbox.area < self.min_crop_area:
                        self.failure_stats["small_detection"] += 1
                    if quality_score < self.crop_quality_threshold:
                        self.failure_stats["low_quality_crop"] += 1
            else:
                self.failure_stats["no_detection_primary"] += 1
                
        except Exception as e:
            self.failure_stats["yolo_error_primary"] += 1
            if self.log_failures:
                logging.warning(f"Primary YOLO failed for {image_path}: {e}")
        
        # Step 2: Fallback YOLO with lower threshold
        try:
            result = self.yolo_fallback.infer_and_get_best_crop(str(image_path))
            
            if result is not None:
                best_bbox, cropped_array = result
                quality_score = self._calculate_crop_quality(cropped_array, best_bbox)
                
                if best_bbox.area >= self.min_crop_area and quality_score >= self.crop_quality_threshold:
                    cropped_image = self._array_to_pil(cropped_array)
                    
                    # Cache with fallback identifier
                    fallback_cache_path = self._get_cache_path(image_path, "yolo_fallback")
                    self._save_to_cache(cropped_image, fallback_cache_path)
                    
                    self.fallback_stats["yolo_fallback_success"] += 1
                    self.crop_quality_stats.append(quality_score)
                    return cropped_image, "yolo_fallback", quality_score
                else:
                    if best_bbox.area < self.min_crop_area:
                        self.failure_stats["small_detection_fallback"] += 1
                    if quality_score < self.crop_quality_threshold:
                        self.failure_stats["low_quality_crop_fallback"] += 1
            else:
                self.failure_stats["no_detection_fallback"] += 1
                
        except Exception as e:
            self.failure_stats["yolo_error_fallback"] += 1
            if self.log_failures:
                logging.warning(f"Fallback YOLO failed for {image_path}: {e}")
        
        # Step 3: Intelligent fallback strategies
        return self._apply_fallback_strategy(image_path)
    
    def _apply_fallback_strategy(self, image_path: Path) -> Tuple[Optional[Image.Image], str, float]:
        """Apply non-YOLO fallback strategy"""
        
        try:
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            
            if self.fallback_strategy == "center_crop":
                processed_image = self._center_crop(original_image)
                method = "center_crop"
                
            elif self.fallback_strategy == "smart_crop":
                processed_image = self._smart_crop(original_image)
                method = "smart_crop"
                
            elif self.fallback_strategy == "edge_detection":
                processed_image = self._edge_detection_crop(original_image)
                method = "edge_detection"
                
            else:
                # Default: simple resize
                processed_image = original_image.resize(self.target_size)
                method = "simple_resize"
            
            # Cache the fallback result
            cache_path = self._get_cache_path(image_path, method)
            self._save_to_cache(processed_image, cache_path)
            
            self.fallback_stats[method] += 1
            return processed_image, method, 0.5  # Moderate quality score for fallbacks
            
        except Exception as e:
            self.failure_stats[f"fallback_error_{self.fallback_strategy}"] += 1
            if self.log_failures:
                logging.error(f"Fallback strategy failed for {image_path}: {e}")
            
            # Last resort: return None and let dataset handle it
            return None, "total_failure", 0.0
    
    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        """Convert NumPy array to PIL Image"""
        if array.dtype == np.uint8:
            return Image.fromarray(array)
        else:
            return Image.fromarray((array * 255).astype(np.uint8))
    
    def _center_crop(self, image: Image.Image) -> Image.Image:
        """Simple center crop strategy"""
        width, height = image.size
        crop_size = min(width, height)
        
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize(self.target_size)
    
    def _smart_crop(self, image: Image.Image) -> Image.Image:
        """Smart crop using image analysis to find interesting regions"""
        
        # Convert to numpy for analysis
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply multiple techniques to find interesting regions
        
        # 1. Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # 2. Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Expand bounding box slightly
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_array.shape[1] - x, w + 2 * margin)
            h = min(img_array.shape[0] - y, h + 2 * margin)
            
            # Make square crop
            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            
            x = max(0, center_x - size // 2)
            y = max(0, center_y - size // 2)
            x = min(img_array.shape[1] - size, x)
            y = min(img_array.shape[0] - size, y)
            
            cropped = image.crop((x, y, x + size, y + size))
            return cropped.resize(self.target_size)
        
        # Fallback to center crop if no contours found
        return self._center_crop(image)
    
    def _edge_detection_crop(self, image: Image.Image) -> Image.Image:
        """Crop using edge detection to find regions of interest"""
        
        # Convert to grayscale
        gray_image = image.convert('L')
        
        # Apply edge detection filter
        edges = gray_image.filter(ImageFilter.FIND_EDGES)
        
        # Convert to numpy for analysis
        edge_array = np.array(edges)
        
        # Find regions with high edge density
        kernel_size = 50
        height, width = edge_array.shape
        
        best_score = 0
        best_region = None
        
        for y in range(0, height - kernel_size, kernel_size // 2):
            for x in range(0, width - kernel_size, kernel_size // 2):
                region = edge_array[y:y+kernel_size, x:x+kernel_size]
                score = np.sum(region)
                
                if score > best_score:
                    best_score = score
                    best_region = (x, y)
        
        if best_region:
            x, y = best_region
            
            # Create square crop around best region
            crop_size = min(kernel_size * 2, min(width - x, height - y))
            cropped = image.crop((x, y, x + crop_size, y + crop_size))
            return cropped.resize(self.target_size)
        
        # Fallback to center crop
        return self._center_crop(image)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get training sample with robust fallback handling"""
        
        image_path, class_idx = self.samples[idx]
        
        # Get processed image with fallback
        processed_image, method_used, quality_score = self._yolo_crop_with_fallback(image_path)
        
        if processed_image is None:
            # Absolute last resort: load original and resize
            try:
                processed_image = Image.open(image_path).convert('RGB')
                processed_image = processed_image.resize(self.target_size)
                method_used = "emergency_resize"
                quality_score = 0.2
                self.fallback_stats["emergency_resize"] += 1
            except Exception as e:
                # Create a black image as absolute fallback
                processed_image = Image.new('RGB', self.target_size, color='black')
                method_used = "black_fallback"
                quality_score = 0.0
                self.failure_stats["total_failure"] += 1
                if self.log_failures:
                    logging.error(f"Total failure for {image_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image_tensor = self.transform(processed_image)
        else:
            image_tensor = transforms.ToTensor()(processed_image)
        
        return image_tensor, class_idx
    
    def get_failure_statistics(self) -> dict:
        """Get detailed failure and fallback statistics"""
        
        total_samples = len(self.samples)
        
        stats = {
            'total_samples': total_samples,
            'failure_counts': dict(self.failure_stats),
            'fallback_counts': dict(self.fallback_stats),
            'failure_rates': {k: v/total_samples for k, v in self.failure_stats.items()},
            'fallback_rates': {k: v/total_samples for k, v in self.fallback_stats.items()},
            'yolo_success_rate': 1 - (self.failure_stats.get('no_detection_primary', 0) / total_samples),
            'total_fallback_rate': sum(self.fallback_stats.values()) / total_samples
        }
        
        # Add quality statistics
        if self.crop_quality_stats:
            stats['crop_quality'] = {
                'mean': np.mean(self.crop_quality_stats),
                'std': np.std(self.crop_quality_stats),
                'min': np.min(self.crop_quality_stats),
                'max': np.max(self.crop_quality_stats),
                'samples': len(self.crop_quality_stats)
            }
        
        return stats
    
    def print_statistics(self):
        """Print detailed failure and fallback statistics"""
        
        stats = self.get_failure_statistics()
        
        print(f"\n📊 YOLODataset Statistics:")
        print(f"Total samples processed: {stats['total_samples']}")
        print(f"YOLO primary success rate: {stats['yolo_success_rate']:.2%}")
        print(f"Total fallback usage rate: {stats['total_fallback_rate']:.2%}")
        print(f"Augmentation mode: {self.augmentation_mode}")
        
        if 'crop_quality' in stats:
            print(f"Crop quality (mean±std): {stats['crop_quality']['mean']:.3f}±{stats['crop_quality']['std']:.3f}")
        
        if stats['failure_counts']:
            print(f"\n❌ Failure breakdown:")
            for failure_type, count in stats['failure_counts'].items():
                rate = count / stats['total_samples']
                print(f"  {failure_type}: {count} ({rate:.2%})")
        
        if stats['fallback_counts']:
            print(f"\n🔄 Fallback usage:")
            for fallback_type, count in stats['fallback_counts'].items():
                rate = count / stats['total_samples']
                print(f"  {fallback_type}: {count} ({rate:.2%})")
    
    def clear_cache(self):
        """Clear the YOLO crop cache"""
        if self.use_cache and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"Cleared cache directory: {self.cache_dir}")


def create_train_val_datasets(
    data_dir: str,
    yolo_model_path: str,
    val_split: float = 0.2,
    train_augmentation: str = "minimal",  # Avoid double augmentation
    val_augmentation: str = "none",       # No augmentation for validation
    **dataset_kwargs
) -> Tuple[YOLODataset, YOLODataset]:
    """
    Create train/val datasets with proper augmentation strategy
    
    Args:
        train_augmentation: Augmentation mode for training ("none", "minimal", "standard")
        val_augmentation: Augmentation mode for validation (should be "none")
    """
    
    # Create full dataset to get samples
    full_dataset = YOLODataset(
        data_dir=data_dir,
        yolo_model_path=yolo_model_path,
        transform=None,
        augmentation_mode="none",  # No transform during sample collection
        **dataset_kwargs
    )
    
    # Split samples maintaining class distribution
    train_samples = []
    val_samples = []
    
    for class_idx in range(len(full_dataset.class_names)):
        class_samples = [(path, idx) for path, idx in full_dataset.samples if idx == class_idx]
        
        if len(class_samples) > 1:
            train_cls, val_cls = train_test_split(
                class_samples, 
                test_size=val_split, 
                random_state=42
            )
        else:
            train_cls, val_cls = class_samples, []
        
        train_samples.extend(train_cls)
        val_samples.extend(val_cls)
    
    # Create separate datasets with different augmentation strategies
    train_dataset = YOLODataset(
        data_dir=data_dir,
        yolo_model_path=yolo_model_path,
        augmentation_mode=train_augmentation,
        **dataset_kwargs
    )
    train_dataset.samples = train_samples
    
    val_dataset = YOLODataset(
        data_dir=data_dir,
        yolo_model_path=yolo_model_path,
        augmentation_mode=val_augmentation,
        **dataset_kwargs
    )
    val_dataset.samples = val_samples
    
    print(f"✓ Train/Val datasets created: {len(train_samples)} train, {len(val_samples)} val")
    print(f"✓ Train augmentation: {train_augmentation}, Val augmentation: {val_augmentation}")
    
    return train_dataset, val_dataset