#!/usr/bin/env python3

from pathlib import Path
from typing import List, Tuple, Dict


def get_image_extensions():
    return [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".tiff"]


def collect_images(images_dir: Path, extensions: List[str] = None) -> List[Path]:
    if extensions is None:
        extensions = get_image_extensions()

    all_images = []
    for ext in extensions:
        all_images.extend(images_dir.glob(f"*{ext}"))
        all_images.extend(images_dir.glob(f"*{ext.upper()}"))

    return sorted(all_images)


def check_annotation(image_path: Path, labels_dir: Path) -> str:
    # Vérifie l'annotation d'une image dans le dossier des labels.
    label_path = labels_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        return 'missing'

    with open(label_path, 'r') as f:
        content = f.read().strip()

    return 'valid' if content else 'empty'


def analyze_dataset(images_dir: str, labels_dir: str, extensions: List[str] = None) -> Dict:
    # Analyse un dataset d'images et leurs annotations.
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        raise ValueError(f"Le dossier images n'existe pas: {images_dir}")
    if not labels_path.exists():
        raise ValueError(f"Le dossier labels n'existe pas: {labels_dir}")

    all_images = collect_images(images_path, extensions)

    results = {
        'total': len(all_images),
        'valid': [],
        'missing': [],
        'empty': [],
        'valid_names': [],
        'missing_names': [],
        'empty_names': []
    }

    for img_path in all_images:
        status = check_annotation(img_path, labels_path)

        if status == 'valid':
            results['valid'].append(img_path)
            results['valid_names'].append(img_path.name)
        elif status == 'missing':
            results['missing'].append(img_path)
            results['missing_names'].append(img_path.name)
        else:  # empty
            results['empty'].append(img_path)
            results['empty_names'].append(img_path.name)

    return results


def save_problematic_list(missing_names: List[str], empty_names: List[str],
                          output_file: str = 'images_problematiques.txt'):
    # Sauvegarde la liste des images problématiques dans un fichier.
    with open(output_file, 'w') as f:
        if missing_names:
            f.write("SANS ANNOTATION:\n")
            for name in missing_names:
                f.write(f"{name}\n")
            f.write("\n")

        if empty_names:
            f.write("ANNOTATION VIDE:\n")
            for name in empty_names:
                f.write(f"{name}\n")


def read_problematic_list(file_path: str = 'images_problematiques.txt') -> Tuple[List[str], List[str]]:
    # Lit la liste des images problématiques depuis un fichier.
    if not Path(file_path).exists():
        return [], []

    sans_annotation = []
    annotation_vide = []
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line == "SANS ANNOTATION:":
                current_section = "sans"
            elif line == "ANNOTATION VIDE:":
                current_section = "vide"
            elif line and current_section:
                if current_section == "sans":
                    sans_annotation.append(line)
                else:
                    annotation_vide.append(line)

    return sans_annotation, annotation_vide


def delete_images(image_names: List[str], images_dir: str,
                  delete_labels: bool = False, labels_dir: str = None) -> Tuple[int, int]:
    # Supprime des images et leurs labels associés.
    images_path = Path(images_dir)
    deleted_images = 0
    deleted_labels = 0

    for img_name in image_names:
        img_path = images_path / img_name
        if img_path.exists():
            img_path.unlink()
            deleted_images += 1

        if delete_labels and labels_dir:
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                label_path.unlink()
                deleted_labels += 1

    return deleted_images, deleted_labels


def print_summary(results: Dict, max_display: int = 5):
    # Affiche un résumé des résultats de l'analyse.
    total = results['total']
    valid = len(results['valid'])
    missing = len(results['missing'])
    empty = len(results['empty'])

    print(f"\nRésumé de l'analyse:")
    print(f"Total d'images: {total}")
    print(f"Images valides: {valid} ({valid / total * 100:.1f}%)")
    print(f"Sans annotation: {missing} ({missing / total * 100:.1f}%)")
    print(f"Annotation vide: {empty} ({empty / total * 100:.1f}%)")

    if results['missing_names']:
        print(f"\nImages sans annotation ({missing}):")
        for name in results['missing_names'][:max_display]:
            print(f"  - {name}")
        if missing > max_display:
            print(f"  ... et {missing - max_display} autres")

    if results['empty_names']:
        print(f"\nImages avec annotation vide ({empty}):")
        for name in results['empty_names'][:max_display]:
            print(f"  - {name}")
        if empty > max_display:
            print(f"  ... et {empty - max_display} autres")