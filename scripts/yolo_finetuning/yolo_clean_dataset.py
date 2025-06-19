#!/usr/bin/env python3

import sys
from pathlib import Path
from yolo_utils import (
    read_problematic_list,
    delete_images
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_dataset.py <images_dir> <labels_dir>")
        print("Exemple: python clean_dataset.py dataset/images/train dataset/labels/train")
        sys.exit(1)

    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]

    # Vérifier les dossiers
    if not Path(images_dir).exists():
        print(f"Erreur: Le dossier {images_dir} n'existe pas.")
        return 1
    if not Path(labels_dir).exists():
        print(f"Erreur: Le dossier {labels_dir} n'existe pas.")
        return 1

    # Lire le fichier de liste
    list_file = 'images_problematiques.txt'
    if not Path(list_file).exists():
        print(f"Erreur: Le fichier {list_file} n'existe pas.")
        print("Lancez d'abord check_annotations.py pour créer ce fichier.")
        return 1

    try:
        # Lire les images problématiques
        missing, empty = read_problematic_list(list_file)

        if not missing and not empty:
            print("Aucune image problématique trouvée dans le fichier.")
            return 0

        # Afficher ce qui sera supprimé
        print(f"Images trouvées dans {list_file}:")
        if missing:
            print(f"  - Sans annotation: {len(missing)}")
        if empty:
            print(f"  - Annotation vide: {len(empty)}")

        total = len(missing) + len(empty)

        # Confirmation
        response = input(f"\nSupprimer {total} images et leurs labels associés? (o/n): ")
        if response.lower() != 'o':
            print("Suppression annulée.")
            return 0

        # Suppression
        print("\nSuppression en cours...")

        # Images sans annotation (pas de labels à supprimer)
        if missing:
            deleted, _ = delete_images(missing, images_dir)
            print(f"Images sans annotation: {deleted} supprimées")

        # Images avec annotation vide (supprimer images ET labels)
        if empty:
            deleted_img, deleted_lbl = delete_images(
                empty,
                images_dir,
                delete_labels=True,
                labels_dir=labels_dir
            )
            print(f"Images avec annotation vide: {deleted_img} images et {deleted_lbl} labels supprimés")

        print("\nNettoyage terminé.")

    except Exception as e:
        print(f"Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())