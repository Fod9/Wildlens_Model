"""
Unified PyTorch inference pipeline: YOLO detection + MobileNet classification
Single framework, clean implementation
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import sys
from typing import Dict, List, Optional, Tuple, Union
import time

# Import YOLO inference
try:
    from scripts.yolo_finetuning.yolo_inference import YOLOInference
    from ..models.mobilenet import WildlifeMobileNet
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure yolo_inference.py and models are available")


class WildlifePipeline:
    """
    Complete wildlife footprint detection and classification pipeline
    using only PyTorch (no TensorFlow dependency)
    
    Pipeline: Image → YOLO Detection → MobileNet Classification
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        mobilenet_model_path: str,
        device: Optional[torch.device] = None,
        yolo_conf_threshold: float = 0.15,  # Optimized default
        yolo_iou_threshold: float = 0.45
    ):
        """
        Args:
            yolo_model_path: Path to trained YOLO model (.pt file)
            mobilenet_model_path: Path to trained MobileNet model (.pth file)
            device: Inference device (auto-detected if None)
            yolo_conf_threshold: YOLO confidence threshold
            yolo_iou_threshold: YOLO IoU threshold
        """
        
        # Setup device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Initializing Wildlife Pipeline on {self.device}")
        
        # Initialize YOLO (PyTorch)
        print("Loading YOLO model...")
        self.yolo = YOLOInference(
            yolo_model_path,
            conf_threshold=yolo_conf_threshold,
            iou_threshold=yolo_iou_threshold
        )
        
        # Load trained MobileNet model (PyTorch)
        print("Loading MobileNet model...")
        self.mobilenet, self.class_names = self._load_mobilenet(mobilenet_model_path)
        self.mobilenet.to(self.device)
        self.mobilenet.eval()
        
        # Setup MobileNet preprocessing transforms
        self.mobilenet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Pipeline initialized successfully!")
        print(f"  - YOLO: {yolo_model_path}")
        print(f"  - MobileNet: {mobilenet_model_path}")
        print(f"  - Classes: {len(self.class_names)} ({self.class_names})")
        print(f"  - Device: {self.device}")
    
    def _load_mobilenet(self, model_path: str) -> Tuple[WildlifeMobileNet, List[str]]:
        """Load trained MobileNet model with metadata"""
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MobileNet model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract metadata
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            num_classes = len(class_names)
        elif 'model_config' in checkpoint and 'num_classes' in checkpoint['model_config']:
            num_classes = checkpoint['model_config']['num_classes']
            class_names = [f"class_{i}" for i in range(num_classes)]  # Fallback names
        else:
            raise ValueError("Could not determine number of classes from model checkpoint")
        
        # Create model
        model = WildlifeMobileNet(num_classes=num_classes, pretrained=False)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)  # Direct state dict
        
        return model, class_names
    
    def _preprocess_crop_for_mobilenet(self, cropped_image: np.ndarray) -> torch.Tensor:
        """Convert YOLO crop to MobileNet input tensor"""
        
        # Convert NumPy array to PIL Image
        if cropped_image.dtype == np.uint8:
            pil_image = Image.fromarray(cropped_image)
        else:
            pil_image = Image.fromarray((cropped_image * 255).astype(np.uint8))
        
        # Apply transforms and add batch dimension
        tensor = self.mobilenet_transform(pil_image)
        batch_tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return batch_tensor.to(self.device)
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Complete pipeline: detect footprint and classify species
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with detection and classification results
        """
        
        start_time = time.time()
        
        print(f"🔍 Processing: {image_path}")
        
        # Stage 1: YOLO Detection
        detection_start = time.time()
        yolo_result = self.yolo.infer_and_get_best_crop(image_path)
        detection_time = time.time() - detection_start
        
        if yolo_result is None:
            return {
                "success": False,
                "error": "No footprints detected",
                "detection_time": detection_time,
                "total_time": time.time() - start_time
            }
        
        best_bbox, cropped_image = yolo_result
        print(f"✓ Footprint detected - Confidence: {best_bbox.confidence:.3f}, Area: {best_bbox.area:.0f}px²")
        
        # Stage 2: MobileNet Classification
        classification_start = time.time()
        
        # Preprocess crop for MobileNet
        input_tensor = self._preprocess_crop_for_mobilenet(cropped_image)
        
        # Run classification
        with torch.no_grad():
            logits = self.mobilenet(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        classification_time = time.time() - classification_start
        total_time = time.time() - start_time
        
        print(f"✓ Species classified: {predicted_class} ({confidence_score:.3f} confidence)")
        
        # Get top-k predictions
        top_k = min(3, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities[0], top_k)
        top_predictions = [
            {
                "class": self.class_names[idx.item()],
                "confidence": prob.item()
            }
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return {
            "success": True,
            "detection": {
                "bbox": best_bbox.bbox,
                "confidence": best_bbox.confidence,
                "area": best_bbox.area,
                "crop_shape": cropped_image.shape
            },
            "classification": {
                "predicted_class": predicted_class,
                "confidence": confidence_score,
                "top_predictions": top_predictions
            },
            "timing": {
                "detection_time": detection_time,
                "classification_time": classification_time,
                "total_time": total_time
            }
        }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Process multiple images"""
        
        results = []
        
        print(f"🔄 Processing batch of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            print(f"\n--- Image {i+1}/{len(image_paths)} ---")
            result = self.predict_single_image(image_path)
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r["success"])
        avg_time = np.mean([r.get("timing", {}).get("total_time", 0) for r in results if r["success"]])
        
        print(f"\n📊 Batch Summary:")
        print(f"  - Successful: {successful}/{len(image_paths)}")
        print(f"  - Average time: {avg_time:.3f}s per image")
        
        return results
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict:
        """
        Process image from NumPy array (useful for video frames or camera input)
        
        Args:
            image_array: Input image as NumPy array (H, W, C)
            
        Returns:
            Dictionary with detection and classification results
        """
        
        start_time = time.time()
        
        # Stage 1: YOLO Detection from array
        detection_start = time.time()
        yolo_result = self.yolo.infer_and_get_best_crop_from_array(image_array)
        detection_time = time.time() - detection_start
        
        if yolo_result is None:
            return {
                "success": False,
                "error": "No footprints detected",
                "detection_time": detection_time,
                "total_time": time.time() - start_time
            }
        
        best_bbox, cropped_image = yolo_result
        
        # Stage 2: MobileNet Classification
        classification_start = time.time()
        
        input_tensor = self._preprocess_crop_for_mobilenet(cropped_image)
        
        with torch.no_grad():
            logits = self.mobilenet(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
        
        classification_time = time.time() - classification_start
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "detection": {
                "bbox": best_bbox.bbox,
                "confidence": best_bbox.confidence,
                "area": best_bbox.area,
                "crop_shape": cropped_image.shape
            },
            "classification": {
                "predicted_class": predicted_class,
                "confidence": confidence_score
            },
            "timing": {
                "detection_time": detection_time,
                "classification_time": classification_time,
                "total_time": total_time
            }
        }
    
    def save_annotated_result(self, image_path: str, result: Dict, output_path: str):
        """Save image with detection and classification annotations"""
        
        if not result["success"]:
            print(f"❌ Cannot annotate failed prediction")
            return
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box
        bbox = result["detection"]["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add classification text
        predicted_class = result["classification"]["predicted_class"]
        confidence = result["classification"]["confidence"]
        
        text = f"{predicted_class}: {confidence:.3f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        
        # Background rectangle for text
        cv2.rectangle(image_rgb, (x1, y1-30), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image_rgb, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Save annotated image
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
        
        print(f"💾 Annotated result saved: {output_path}")


def load_pipeline(
    yolo_model_path: str = "notebooks/yolo/best_so_far.pt",
    mobilenet_model_path: str = "models/claude_checkpoints/wildlife_mobilenet_final.pth"
) -> WildlifePipeline:
    """
    Factory function to create wildlife pipeline
    
    Args:
        yolo_model_path: Path to YOLO model
        mobilenet_model_path: Path to trained PyTorch MobileNet
        
    Returns:
        WildlifePipeline instance
    """
    return WildlifePipeline(yolo_model_path, mobilenet_model_path)