#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse


def read_image_list(file_path, section=None):
    # Lit le fichier et retourne les listes d'images sans annotation et avec annotation vide
    if not os.path.exists(file_path):
        print(f"Erreur: Le fichier {file_path} n'existe pas.")
        return None, None

    sans_annotation = []
    annotation_vide = []
    current_section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line == "SANS ANNOTATION:":
                current_section = "sans"
                continue
            elif line == "ANNOTATION VIDE:":
                current_section = "vide"
                continue
            elif line == "":
                continue

            if current_section == "sans":
                sans_annotation.append(line)
            elif current_section == "vide":
                annotation_vide.append(line)

    if section == "sans":
        return sans_annotation, []
    elif section == "vide":
        return [], annotation_vide
    else:
        return sans_annotation, annotation_vide


def delete_files(images_list, images_dir, labels_dir, delete_labels=True):
    # Supprime les images et optionnellement leurs labels
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    deleted_images = 0
    deleted_labels = 0

    for img_name in images_list:
        # Supprimer l'image
        img_path = images_path / img_name
        if img_path.exists():
            img_path.unlink()
            deleted_images += 1

        # Supprimer le label si demandé
        if delete_labels:
            label_name = img_path.stem + '.txt'
            label_path = labels_path / label_name
            if label_path.exists():
                label_path.unlink()
                deleted_labels += 1

    return deleted_images, deleted_labels


def main():
    parser = argparse.ArgumentParser(description='Supprimer des images depuis une liste')
    parser.add_argument('images_dir', help='Dossier des images')
    parser.add_argument('labels_dir', help='Dossier des labels')
    parser.add_argument('--file', default='images_problematiques.txt', help='Fichier de liste')
    parser.add_argument('--type', choices=['sans', 'vide', 'tous'], default='tous',
                        help='Type d\'images à supprimer: sans annotation, annotation vide, ou tous')
    parser.add_argument('--force', action='store_true', help='Supprimer sans confirmation')

    args = parser.parse_args()

    # Vérifier les dossiers
    if not os.path.exists(args.images_dir):
        print(f"Erreur: Le dossier {args.images_dir} n'existe pas.")
        sys.exit(1)
    if not os.path.exists(args.labels_dir):
        print(f"Erreur: Le dossier {args.labels_dir} n'existe pas.")
        sys.exit(1)

    # Lire les listes
    sans, vide = read_image_list(args.file, args.type if args.type != 'tous' else None)

    if sans is None and vide is None:
        sys.exit(1)

    # Afficher le résumé
    total = 0
    if sans and args.type in ['sans', 'tous']:
        print(f"Images sans annotation: {len(sans)}")
        total += len(sans)
    if vide and args.type in ['vide', 'tous']:
        print(f"Images avec annotation vide: {len(vide)}")
        total += len(vide)

    if total == 0:
        print("Aucune image à supprimer.")
        return

    # Confirmation
    if not args.force:
        response = input(f"\nSupprimer {total} images? (o/n): ")
        if response.lower() != 'o':
            print("Annulé.")
            return

    # Suppression
    print("\nSuppression...")

    if sans and args.type in ['sans', 'tous']:
        del_img, del_lbl = delete_files(sans, args.images_dir, args.labels_dir, delete_labels=False)
        print(f"Sans annotation: {del_img} images supprimées")

    if vide and args.type in ['vide', 'tous']:
        del_img, del_lbl = delete_files(vide, args.images_dir, args.labels_dir, delete_labels=True)
        print(f"Annotation vide: {del_img} images et {del_lbl} labels supprimés")

    print("\nTerminé.")


if __name__ == "__main__":
    main()