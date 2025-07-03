from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
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
    def get_areas_from_bboxes(bboxes: List[List[float]]) -> List[float]:
        """Calcule les aires à partir d'une liste de bboxes [x1, y1, x2, y2]"""
        areas = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
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
    def get_bboxes_with_scores(detections: List[dict]) -> Optional[List[BBoxWithScore]]:
        """Convertit les détections ONNX en BBoxWithScore"""
        if not detections:
            return None

        bboxes = [det['bbox'] for det in detections]
        confidences = [det['confidence'] for det in detections]

        areas = AreaUtility.get_areas_from_bboxes(bboxes)
        scores = AreaUtility.calculate_bbox_scores(areas, confidences)

        bboxes_with_scores = []
        for bbox, score, confidence, area in zip(bboxes, scores, confidences, areas):
            bbox_with_score = BBoxWithScore(
                bbox=bbox,
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


def postprocess_yolo_output(output: np.ndarray, conf_threshold: float = 0.25,
                            iou_threshold: float = 0.45) -> List[dict]:
    """
    Post-traite la sortie brute ONNX pour extraire les bounding boxes

    Args:
        output: Sortie ONNX de forme (1, 5, 8400) ou (1, 4+num_classes, 8400)
        conf_threshold: Seuil de confiance minimum
        iou_threshold: Seuil IoU pour la suppression non-maximale

    Returns:
        Liste de dictionnaires contenant les détections
    """
    predictions = output[0].T  # Shape: (8400, 5)

    detections = []

    for detection in predictions:
        if len(detection) == 5:
            x_center, y_center, width, height, confidence = detection
            class_id = 0
        else:
            x_center, y_center, width, height = detection[:4]
            class_scores = detection[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

        if confidence < conf_threshold:
            continue

        # Convertir de center format vers corner format
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': float(confidence),
            'class_id': int(class_id),
            'x_center': float(x_center),
            'y_center': float(y_center),
            'width': float(width),
            'height': float(height)
        })

    # Appliquer NMS
    if detections:
        detections = apply_nms(detections, iou_threshold)

    return detections


def apply_nms(detections: List[dict], iou_threshold: float) -> List[dict]:
    """Applique la suppression non-maximale"""
    if not detections:
        return []

    boxes = [det['bbox'] for det in detections]
    scores = [det['confidence'] for det in detections]

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)

    if len(indices) > 0:
        indices = indices.flatten()
        return [detections[i] for i in indices]
    else:
        return []


class YOLOONNXInference:
    """Classe d'inférence YOLO avec ONNX - Interface compatible avec l'ancienne classe"""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Obtenir les dimensions d'entrée
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Préprocesse l'image pour l'inférence ONNX"""
        original_shape = image.shape[:2]  # (height, width)

        # Redimensionner l'image
        img_resized = cv2.resize(image, (self.input_width, self.input_height))

        # Normaliser et convertir en format ONNX (NCHW)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_transposed, axis=0)  # Ajouter dimension batch

        return img_batch, original_shape

    def _load_image_from_source(self, source: ImageType) -> np.ndarray:
        """Charge une image depuis différents types de sources"""
        if isinstance(source, str) or isinstance(source, Path):
            # Chemin vers fichier image
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {source}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        elif isinstance(source, Image.Image):
            # Image PIL
            return np.array(source.convert('RGB'))

        elif isinstance(source, np.ndarray):
            # Array numpy
            if len(source.shape) == 3 and source.shape[2] == 3:
                return source
            else:
                raise ValueError("L'array numpy doit avoir la forme (H, W, 3)")

        else:
            raise ValueError(f"Type de source non supporté: {type(source)}")

    def _scale_detections_to_original(self, detections: List[dict],
                                      original_shape: Tuple[int, int]) -> List[dict]:
        """Redimensionne les détections vers la taille originale de l'image"""
        orig_h, orig_w = original_shape
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ]

            # Mettre à jour les autres coordonnées si nécessaire
            det['x_center'] *= scale_x
            det['y_center'] *= scale_y
            det['width'] *= scale_x
            det['height'] *= scale_y

        return detections

    def predict(self, source: ImageType, save_conf: bool = True) -> List[dict]:
        """
        Effectue la prédiction - Interface compatible avec l'ancienne méthode

        Args:
            source: Source de l'image (chemin, PIL Image, ou numpy array)
            save_conf: Compatibilité avec l'ancienne interface (ignoré)

        Returns:
            Liste des détections au format compatible
        """
        # Charger et préprocesser l'image
        image = self._load_image_from_source(source)
        input_data, original_shape = self._preprocess_image(image)

        # Inférence ONNX
        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        # Post-traitement
        detections = postprocess_yolo_output(
            outputs[0],
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )

        # Redimensionner vers la taille originale
        detections = self._scale_detections_to_original(detections, original_shape)

        return detections

    def infer_and_get_best_crop(self, image_path: str) -> Optional[tuple[BBoxWithScore, ndarray]]:
        """Effectue l'inférence et retourne le meilleur crop"""
        # Perform inference
        detections = self.predict(image_path)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(detections)

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
        """Effectue l'inférence à partir d'un array et retourne le meilleur crop"""
        # Perform inference
        detections = self.predict(image_array)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(detections)

        if bboxes_with_scores is None:
            return None

        # Get best bbox
        best_bbox = BBoxUtility.get_best_bbox(bboxes_with_scores)

        if best_bbox is None:
            return None

        # Convertir array en PIL pour le crop
        pil_image = Image.fromarray(image_array)
        cropped_image = BBoxUtility.crop_image_from_pil(pil_image, best_bbox.bbox)

        return best_bbox, cropped_image

    def get_all_crops(self, image_path: str) -> List[Tuple[BBoxWithScore, np.ndarray]]:
        """Retourne tous les crops détectés"""
        # Perform inference
        detections = self.predict(image_path)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(detections)

        if bboxes_with_scores is None:
            return []

        crops = []
        for bbox_with_score in bboxes_with_scores:
            cropped_image = BBoxUtility.crop_image_from_bbox(image_path, bbox_with_score.bbox)
            crops.append((bbox_with_score, cropped_image))

        return crops

    def get_all_crops_from_array(self, image_array: np.ndarray) -> List[Tuple[BBoxWithScore, np.ndarray]]:
        """Retourne tous les crops détectés à partir d'un array"""
        # Perform inference
        detections = self.predict(image_array)

        # Get bboxes with scores
        bboxes_with_scores = BBoxUtility.get_bboxes_with_scores(detections)

        if bboxes_with_scores is None:
            return []

        pil_image = Image.fromarray(image_array)
        crops = []
        for bbox_with_score in bboxes_with_scores:
            cropped_image = BBoxUtility.crop_image_from_pil(pil_image, bbox_with_score.bbox)
            crops.append((bbox_with_score, cropped_image))

        return crops

if __name__ == "__main__":
    # Initialize the inference class - REMPLACEZ PAR VOTRE MODÈLE ONNX
    yolo_inference = YOLOONNXInference('best_so_far.onnx', conf_threshold=0.25, iou_threshold=0.45)

    # Test avec une image
    image_array = cv2.imread('test_images/footprint_1.jpeg')
    if image_array is not None:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Test de la méthode principale
        result_from_array = yolo_inference.infer_and_get_best_crop_from_array(image_array)
        if result_from_array is not None:
            best_bbox, cropped_image = result_from_array
            print(f"From array - Score: {best_bbox.score:.4f}, Shape: {cropped_image.shape}")
            print(f"Best BBox: {best_bbox.bbox}, Confidence: {best_bbox.confidence:.4f}, Area: {best_bbox.area:.2f}")

            # Sauvegarder le résultat
            output_path = 'test_images/result_from_array_onnx.jpg'
            cv2.imwrite(output_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
            print(f"Cropped image saved to: {output_path}")
        else:
            print("No bounding boxes found in the image.")

        # Test pour obtenir tous les crops
        all_crops = yolo_inference.get_all_crops_from_array(image_array)
        print(f"Found {len(all_crops)} total detections")

        for i, (bbox_info, crop) in enumerate(all_crops):
            output_path = f'test_images/crop_{i}_onnx.jpg'
            cv2.imwrite(output_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            print(f"Crop {i}: Score={bbox_info.score:.4f}, Confidence={bbox_info.confidence:.4f}")
    else:
        print("Could not load test image")