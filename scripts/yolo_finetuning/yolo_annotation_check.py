#!/usr/bin/env python3

from pathlib import Path


def check_yolo_annotations(images_dir: str, labels_dir: str) -> tuple[int, int, int]:
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    # Extensions d'images supportées
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Collecter toutes les images
    all_images = []
    for ext in image_extensions:
        all_images.extend(images_path.glob(f"*{ext}"))

    # Compteurs
    total = len(all_images)
    missing = []
    empty = []
    valid = []

    print(f"Vérification de {total} images...\n")

    for img in all_images:
        label_file = labels_path / f"{img.stem}.txt"

        if not label_file.exists():
            missing.append(img.name)
        elif label_file.stat().st_size == 0:
            empty.append(img.name)
        else:
            # Vérifier qu'il y a au moins une ligne valide
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    valid.append(img.name)
                else:
                    empty.append(img.name)

    # Afficher le résumé
    print(f"Images avec annotations valides: {len(valid)} ({len(valid) / total * 100:.1f}%)")
    print(f"Images sans fichier d'annotation: {len(missing)} ({len(missing) / total * 100:.1f}%)")
    print(f"Images avec annotations vides: {len(empty)} ({len(empty) / total * 100:.1f}%)")

    # Afficher les fichiers problématiques
    if missing:
        print(f"\nImages sans annotation ({len(missing)}):")
        for img in missing[:5]:
            print(f"   - {img}")
        if len(missing) > 5:
            print(f"   ... et {len(missing) - 5} autres")

    if empty:
        print(f"\nImages avec annotation vide ({len(empty)}):")
        for img in empty[:5]:
            print(f"   - {img}")
        if len(empty) > 5:
            print(f"   ... et {len(empty) - 5} autres")

    # Proposer de sauvegarder la liste
    if missing or empty:
        save = input("\nVoulez-vous sauvegarder la liste des images problématiques? (o/n): ")
        if save.lower() == 'o':
            with open('images_problematiques.txt', 'w') as f:
                if missing:
                    f.write("SANS ANNOTATION:\n")
                    for img in missing:
                        f.write(f"{img}\n")
                    f.write("\n")
                if empty:
                    f.write("ANNOTATION VIDE:\n")
                    for img in empty:
                        f.write(f"{img}\n")
            print("Liste sauvegardée dans: images_problematiques.txt")

    return len(valid), len(missing), len(empty)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python yolo_annotation_check.py <images_dir> <labels_dir>")
        sys.exit(1)

    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]

    check_yolo_annotations(images_dir, labels_dir)