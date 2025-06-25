from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
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


class AreaUtility:

    @staticmethod
    def calculate_bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return width * height

    @staticmethod
    def get_areas_from_bboxes(bboxes: Boxes) -> List[float]:
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
    def normalize_bbox(bbox: torch.Tensor) -> List[float]:
        return [float(coord) for coord in bbox]

    @staticmethod
    def get_bboxes_with_scores(bboxes: Boxes) -> Optional[List[BBoxWithScore]]:
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


class YOLOInference:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, source: ImageType, save_conf: bool = True) -> Results:
        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save_conf=save_conf,
        )
        return results[0]

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
        results = self.predict_from_array(image_array)

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
    # Initialize the inference class
    yolo_inference = YOLOInference('best_so_far.pt', conf_threshold=0.25, iou_threshold=0.45)

    image_array = cv2.imread('test_images/footprint_1.jpeg')
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    result_from_array = yolo_inference.infer_and_get_best_crop_from_array(image_array)
    if result_from_array is not None:
        best_bbox, cropped_image = result_from_array
        print(f"From array - Score: {best_bbox.score:.4f}, Shape: {cropped_image.shape}")
        print(f"Best BBox: {best_bbox.bbox}, Confidence: {best_bbox.confidence:.4f}, Area: {best_bbox.area:.2f}")
        # Display the cropped image
        cv2.imwrite('test_images/result_from_array.jpg', cropped_image)
    else:
        print("No bounding boxes found in the image.")


