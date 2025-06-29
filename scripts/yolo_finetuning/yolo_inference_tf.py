from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from PIL import Image
import cv2
from typing import Union
from numpy import ndarray

ImageType = Union[str, Path, Image.Image, ndarray]


@dataclass
class BBoxWithScore:
    bbox: List[float]
    score: float
    confidence: float
    area: float


class TensorFlowResults:
    """TensorFlow equivalent of Ultralytics Results class."""
    
    def __init__(self, boxes: Optional['TensorFlowBoxes'] = None):
        self.boxes = boxes


class TensorFlowBoxes:
    """TensorFlow equivalent of Ultralytics Boxes class."""
    
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray):
        self.xyxy = xyxy  # Shape: (N, 4) - bounding boxes in xyxy format
        self.conf = conf  # Shape: (N,) - confidence scores


class AreaUtility:

    @staticmethod
    def calculate_bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height

    @staticmethod
    def get_areas_from_bboxes(bboxes: TensorFlowBoxes) -> List[float]:
        areas = []
        for bbox in bboxes.xyxy:
            bbox_coords = [float(coord) for coord in bbox]
            x1, y1, x2, y2 = bbox_coords
            area = AreaUtility.calculate_bbox_area(x1, y1, x2, y2)
            areas.append(area)
        return areas

    @staticmethod
    def calculate_bbox_scores(areas: List[float], confidences: List[float]) -> List[float]:
        if len(areas) != len(confidences):
            raise ValueError("Areas and confidences must have the same length.")

        scores = [area * confidence for area, confidence in zip(areas, confidences)]
        return scores


class BBoxUtility:

    @staticmethod
    def normalize_bbox(bbox: np.ndarray) -> List[float]:
        return [float(coord) for coord in bbox]

    @staticmethod
    def get_bboxes_with_scores(bboxes: TensorFlowBoxes) -> Optional[List[BBoxWithScore]]:
        if bboxes is None or len(bboxes.xyxy) == 0:
            return None

        areas = AreaUtility.get_areas_from_bboxes(bboxes)
        confidences = bboxes.conf.tolist()
        scores = AreaUtility.calculate_bbox_scores(areas, confidences)

        bboxes_with_scores = []
        for bbox, score, confidence, area in zip(bboxes.xyxy, scores, confidences, areas):
            bbox_with_score = BBoxWithScore(
                bbox=bbox.tolist(),
                score=score,
                confidence=confidence,
                area=area
            )
            bboxes_with_scores.append(bbox_with_score)

        return bboxes_with_scores

    @staticmethod
    def get_best_bbox(bboxes_with_scores: List[BBoxWithScore]) -> Optional[BBoxWithScore]:
        if not bboxes_with_scores:
            return None
        return max(bboxes_with_scores, key=lambda x: x.score)

    @staticmethod
    def crop_image_from_bbox(image_path: str, bbox: List[float]) -> np.ndarray:
        img = Image.open(image_path)
        x1, y1, x2, y2 = map(int, bbox[:4])
        cropped_image = img.crop((x1, y1, x2, y2))
        return np.array(cropped_image)

    @staticmethod
    def crop_image_from_pil(pil_image: Image.Image, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cropped_image = pil_image.crop((x1, y1, x2, y2))
        return np.array(cropped_image)


class YOLOInferenceTF:
    """TensorFlow-compatible YOLO inference using ONNX Runtime."""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        # Support both .pt (will look for .onnx) and .onnx files
        if model_path.endswith('.pt'):
            # Look for corresponding ONNX file
            onnx_path = model_path.replace('.pt', '.onnx')
            if not Path(onnx_path).exists():
                raise ValueError(f"ONNX model not found: {onnx_path}. Please convert PyTorch model to ONNX first.")
            model_path = onnx_path
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input shape info
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]  # Assuming NCHW format
        self.input_width = input_shape[3]
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image array in NCHW format
        """
        
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed (OpenCV format)
            if len(image.shape) == 3 and image.shape[2] == 3:
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize to model input size
        img = img.resize((self.input_width, self.input_height), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Convert HWC to CHW
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def postprocess_output(self, output: np.ndarray, original_image_size: Tuple[int, int] = None) -> TensorFlowBoxes:
        """
        Post-process YOLO output to extract bounding boxes and confidences.
        
        Args:
            output: Raw YOLO output array, shape (1, 5, 8400)
            original_image_size: Optional original image size for scaling
            
        Returns:
            TensorFlowBoxes object with filtered detections
        """
        
        # Output format: [batch, 5, num_anchors] where 5 = [x_center, y_center, width, height, confidence]
        output = output[0]  # Remove batch dimension -> (5, 8400)
        output = output.T   # Transpose to (8400, 5)
        
        # Extract components
        x_center = output[:, 0]
        y_center = output[:, 1]
        width = output[:, 2]
        height = output[:, 3]
        confidence = output[:, 4]
        
        # Filter by confidence threshold
        mask = confidence >= self.conf_threshold
        if not np.any(mask):
            return TensorFlowBoxes(np.empty((0, 4)), np.empty((0,)))
        
        # Apply mask
        x_center = x_center[mask]
        y_center = y_center[mask]
        width = width[mask]
        height = height[mask]
        confidence = confidence[mask]
        
        # Convert center format to xyxy format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Stack to create bounding boxes
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS (Non-Maximum Suppression)
        boxes, confidence = self.apply_nms(boxes, confidence)
        
        return TensorFlowBoxes(boxes, confidence)

    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression using TensorFlow operations.
        
        Args:
            boxes: Bounding boxes in xyxy format, shape (N, 4)
            scores: Confidence scores, shape (N,)
            
        Returns:
            Filtered boxes and scores
        """
        
        if len(boxes) == 0:
            return boxes, scores
        
        # Use TensorFlow's NMS implementation
        indices = tf.image.non_max_suppression(
            boxes=tf.constant(boxes, dtype=tf.float32),
            scores=tf.constant(scores, dtype=tf.float32),
            max_output_size=100,  # Maximum number of boxes to keep
            iou_threshold=self.iou_threshold
        )
        
        # Convert back to numpy
        indices = indices.numpy()
        
        return boxes[indices], scores[indices]

    def predict(self, source: ImageType, save_conf: bool = True) -> TensorFlowResults:
        """
        Run YOLO inference on an image.
        
        Args:
            source: Input image (file path, numpy array, or PIL Image)
            save_conf: Not used, kept for compatibility
            
        Returns:
            TensorFlowResults object containing detection results
        """
        
        # Preprocess image
        input_array = self.preprocess_image(source)
        
        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        output = outputs[0]  # Get the first (and only) output
        
        # Post-process to get bounding boxes
        boxes = self.postprocess_output(output)
        
        return TensorFlowResults(boxes)

    def infer_and_get_best_crop(self, image_path: str) -> Optional[Tuple[BBoxWithScore, np.ndarray]]:
        # Perform inference
        results = self.predict(image_path)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(results.boxes)

        if bboxes_with_scores is None:
            return None

        # Get best bbox
        best_bbox = BBoxUtility.get_best_bbox(bboxes_with_scores)

        if best_bbox is None:
            return None

        # Crop image
        cropped_image = BBoxUtility.crop_image_from_bbox(image_path, best_bbox.bbox)

        return best_bbox, cropped_image

    def infer_and_get_best_crop_from_array(self, image_array: np.ndarray) -> Optional[Tuple[BBoxWithScore, np.ndarray]]:
        # Perform inference
        results = self.predict(image_array)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(results.boxes)

        if bboxes_with_scores is None:
            return None

        # Get best bbox
        best_bbox = BBoxUtility.get_best_bbox(bboxes_with_scores)

        if best_bbox is None:
            return None

        pil_image = Image.fromarray(image_array)
        cropped_image = BBoxUtility.crop_image_from_pil(pil_image, best_bbox.bbox)

        return best_bbox, cropped_image

    def get_all_crops(self, image_path: str) -> List[Tuple[BBoxWithScore, np.ndarray]]:
        # Perform inference
        results = self.predict(image_path)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(results.boxes)

        if bboxes_with_scores is None:
            return []

        crops = []
        for bbox_with_score in bboxes_with_scores:
            cropped_image = BBoxUtility.crop_image_from_bbox(image_path, bbox_with_score.bbox)
            crops.append((bbox_with_score, cropped_image))

        return crops

    def get_all_crops_from_array(self, image_array: np.ndarray) -> List[Tuple[BBoxWithScore, np.ndarray]]:
        # Perform inference
        results = self.predict(image_array)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(results.boxes)

        if bboxes_with_scores is None:
            return []

        pil_image = Image.fromarray(image_array)
        crops = []
        for bbox_with_score in bboxes_with_scores:
            cropped_image = BBoxUtility.crop_image_from_pil(pil_image, bbox_with_score.bbox)
            crops.append((bbox_with_score, cropped_image))

        return crops


if __name__ == "__main__":
    # Initialize the TensorFlow-compatible inference class
    yolo_inference = YOLOInferenceTF('notebooks/yolo/best_so_far.pt', conf_threshold=0.25, iou_threshold=0.45)

    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    result_from_array = yolo_inference.infer_and_get_best_crop_from_array(dummy_image)
    if result_from_array is not None:
        best_bbox, cropped_image = result_from_array
        print(f"From array - Score: {best_bbox.score:.4f}, Shape: {cropped_image.shape}")
        print(f"Best BBox: {best_bbox.bbox}, Confidence: {best_bbox.confidence:.4f}, Area: {best_bbox.area:.2f}")
    else:
        print("No bounding boxes found in the image.")