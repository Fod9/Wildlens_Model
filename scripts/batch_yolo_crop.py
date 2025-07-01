#!/usr/bin/env python3
"""
Script de cropping en batch utilisant YOLO

Ce script traite toutes les images d'un dossier d'entrée en utilisant YOLO pour détecter
et cropper les empreintes d'animaux, puis sauvegarde les résultats dans un dossier de sortie
en préservant l'architecture des dossiers.

Usage:
    python batch_yolo_crop.py <input_folder> <output_folder> [--model_path] [--conf_threshold] [--fallback]

Exemples:
    python batch_yolo_crop.py ./data/raw ./data/cropped
    python batch_yolo_crop.py ./data/raw ./data/cropped --conf_threshold 0.3 --fallback center_crop
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes


class BatchYOLOCropper:
    """Classe pour traiter des images en batch avec YOLO cropping."""
    
    def __init__(
        self, 
        model_path: str, 
        conf_threshold: float = 0.25, 
        iou_threshold: float = 0.45,
        fallback_strategy: str = "center_crop"
    ):
        """
        Initialise le cropper YOLO.
        
        Args:
            model_path: Chemin vers le modèle YOLO (.pt)
            conf_threshold: Seuil de confiance pour les détections
            iou_threshold: Seuil IoU pour le NMS
            fallback_strategy: Stratégie de fallback ('center_crop', 'skip', 'original')
        """
        # Charger le modèle YOLO directement avec ultralytics
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.fallback_strategy = fallback_strategy
        
        # Statistiques
        self.stats = {
            "total_images": 0,
            "successful_crops": 0,
            "failed_detections": 0,
            "fallback_used": 0,
            "errors": 0
        }
        
        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Extensions d'images supportées
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def _get_image_files(self, input_dir: Path) -> list:
        """Récupère tous les fichiers d'images dans le dossier et ses sous-dossiers."""
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if Path(file).suffix.lower() in self.image_extensions:
                    image_files.append(Path(root) / file)
        return image_files
    
    def _create_output_path(self, input_path: Path, input_dir: Path, output_dir: Path) -> Path:
        """Crée le chemin de sortie en préservant l'architecture des dossiers."""
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _apply_fallback_strategy(self, image_path: Path) -> Optional[Image.Image]:
        """Applique la stratégie de fallback si YOLO échoue."""
        if self.fallback_strategy == "skip":
            return None
        elif self.fallback_strategy == "original":
            return Image.open(image_path)
        elif self.fallback_strategy == "center_crop":
            return self._center_crop(image_path)
        else:
            self.logger.warning(f"Stratégie de fallback inconnue: {self.fallback_strategy}")
            return None
    
    def _center_crop(self, image_path: Path) -> Image.Image:
        """Effectue un crop centré carré de l'image."""
        img = Image.open(image_path)
        width, height = img.size
        
        # Créer un crop carré centré
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        cropped_img = img.crop((left, top, right, bottom))
        return cropped_img
    
    def _get_best_crop_from_results(self, results: List[Results], image_path: Path) -> Optional[Image.Image]:
        """
        Extrait le meilleur crop des résultats YOLO.
        
        Returns:
            PIL Image du crop ou None si aucune détection
        """
        if not results or len(results) == 0:
            return None
            
        result = results[0]  # Premier résultat
        if result.boxes is None or len(result.boxes) == 0:
            return None
            
        # Prendre la bbox avec la meilleure confiance
        best_box_idx = 0
        if len(result.boxes.conf) > 1:
            best_box_idx = result.boxes.conf.argmax().item()
        
        # Extraire les coordonnées de la bbox
        box = result.boxes.xyxy[best_box_idx]
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Charger l'image et cropper
        img = Image.open(image_path)
        cropped_image = img.crop((x1, y1, x2, y2))
        
        return cropped_image
    
    def _process_single_image(self, image_path: Path, output_path: Path) -> bool:
        """
        Traite une seule image.
        
        Returns:
            bool: True si le traitement a réussi, False sinon
        """
        try:
            self.stats["total_images"] += 1
            
            # Tentative de détection et crop avec YOLO
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save_conf=True,
                verbose=False  # Désactiver les prints
            )
            
            # Essayer d'extraire le meilleur crop
            cropped_image = self._get_best_crop_from_results(results, image_path)
            
            if cropped_image is not None:
                # YOLO a réussi
                processed_image = cropped_image
                self.stats["successful_crops"] += 1
            else:
                # YOLO a échoué - utiliser fallback
                processed_image = self._apply_fallback_strategy(image_path)
                self.stats["failed_detections"] += 1
                if processed_image is not None:
                    self.stats["fallback_used"] += 1
            
            # Sauvegarder l'image traitée
            if processed_image is not None:
                # Convertir en PIL Image si c'est un numpy array
                if isinstance(processed_image, np.ndarray):
                    if processed_image.dtype == np.uint8:
                        pil_image = Image.fromarray(processed_image)
                    else:
                        # Normaliser si les valeurs sont en float
                        processed_image = (processed_image * 255).astype(np.uint8)
                        pil_image = Image.fromarray(processed_image)
                else:
                    pil_image = processed_image
                
                # Sauvegarder avec la même extension ou forcer en JPEG
                pil_image.save(output_path, quality=95)
                return True
            else:
                self.logger.warning(f"Impossible de traiter l'image: {image_path}")
                return False
                
        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Erreur lors du traitement de {image_path}: {e}")
            return False
    
    def process_folder(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> dict:
        """
        Traite toutes les images d'un dossier.
        
        Args:
            input_dir: Dossier d'entrée contenant les images
            output_dir: Dossier de sortie pour les images croppées
            
        Returns:
            dict: Statistiques du traitement
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise ValueError(f"Le dossier d'entrée n'existe pas: {input_dir}")
        
        # Créer le dossier de sortie
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Récupérer tous les fichiers d'images
        image_files = self._get_image_files(input_dir)
        
        if not image_files:
            self.logger.warning(f"Aucune image trouvée dans {input_dir}")
            return self.stats
        
        self.logger.info(f"Traitement de {len(image_files)} images...")
        
        # Traiter chaque image avec une barre de progression
        for image_path in tqdm(image_files, desc="Traitement des images"):
            output_path = self._create_output_path(image_path, input_dir, output_dir)
            self._process_single_image(image_path, output_path)
        
        # Afficher les statistiques finales
        self._print_statistics()
        
        return self.stats
    
    def _print_statistics(self):
        """Affiche les statistiques du traitement."""
        total = self.stats["total_images"]
        successful = self.stats["successful_crops"]
        fallback = self.stats["fallback_used"]
        failed = self.stats["failed_detections"]
        errors = self.stats["errors"]
        
        self.logger.info("\n" + "="*50)
        self.logger.info("STATISTIQUES DU TRAITEMENT")
        self.logger.info("="*50)
        self.logger.info(f"Images totales traitées: {total}")
        self.logger.info(f"Crops YOLO réussis: {successful} ({successful/total*100:.1f}%)")
        self.logger.info(f"Détections échouées: {failed} ({failed/total*100:.1f}%)")
        self.logger.info(f"Fallback utilisés: {fallback} ({fallback/total*100:.1f}%)")
        self.logger.info(f"Erreurs: {errors} ({errors/total*100:.1f}%)")
        self.logger.info(f"Taux de succès global: {(successful + fallback)/total*100:.1f}%")
        self.logger.info("="*50)


def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Script de cropping en batch utilisant YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'usage:
  python batch_yolo_crop.py ./data/raw ./data/cropped
  python batch_yolo_crop.py ./data/raw ./data/cropped --conf_threshold 0.3
  python batch_yolo_crop.py ./data/raw ./data/cropped --fallback skip
        """
    )
    
    parser.add_argument(
        "input_folder",
        type=str,
        help="Dossier d'entrée contenant les images à traiter"
    )
    
    parser.add_argument(
        "output_folder", 
        type=str,
        help="Dossier de sortie pour les images croppées"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo_finetuning/best_so_far.pt",
        help="Chemin vers le modèle YOLO (défaut: yolo_finetuning/best_so_far.pt)"
    )
    
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Seuil de confiance pour les détections YOLO (défaut: 0.25)"
    )
    
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.45,
        help="Seuil IoU pour le NMS (défaut: 0.45)"
    )
    
    parser.add_argument(
        "--fallback",
        type=str,
        choices=["center_crop", "skip", "original"],
        default="center_crop",
        help="Stratégie de fallback si YOLO échoue (défaut: center_crop)"
    )
    
    args = parser.parse_args()
    
    # Vérifier que le modèle existe
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        # Chemin relatif au script
        model_path = Path(__file__).parent / model_path
    
    if not model_path.exists():
        print(f"Erreur: Le modèle YOLO n'existe pas: {model_path}")
        sys.exit(1)
    
    # Créer et configurer le cropper
    try:
        cropper = BatchYOLOCropper(
            model_path=str(model_path),
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            fallback_strategy=args.fallback
        )
        
        print(f"Début du traitement...")
        print(f"Dossier d'entrée: {args.input_folder}")
        print(f"Dossier de sortie: {args.output_folder}")
        print(f"Modèle YOLO: {model_path}")
        print(f"Seuil de confiance: {args.conf_threshold}")
        print(f"Stratégie de fallback: {args.fallback}")
        print("-" * 50)
        
        # Traiter le dossier
        stats = cropper.process_folder(args.input_folder, args.output_folder)
        
        print("\nTraitement terminé avec succès!")
        
    except Exception as e:
        print(f"Erreur lors du traitement: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 