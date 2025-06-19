#!/usr/bin/env python3

import sys

from yolo_cleaning_util import (
    analyze_dataset,
    save_problematic_list,
    print_summary
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python check_annotations.py <images_dir> <labels_dir>")
        print("Exemple: python check_annotations.py dataset/images/train dataset/labels/train")
        sys.exit(1)

    images_dir = sys.argv[1]
    labels_dir = sys.argv[2]

    try:
        # Analyser le dataset
        print("Analyse du dataset en cours...")
        results = analyze_dataset(images_dir, labels_dir)

        # Afficher le résumé
        print_summary(results)

        # Proposer de sauvegarder si des problèmes sont trouvés
        if results['missing_names'] or results['empty_names']:
            response = input("\nSauvegarder la liste des images problématiques? (o/n): ")
            if response.lower() == 'o':
                save_problematic_list(
                    results['missing_names'],
                    results['empty_names']
                )
                print("Liste sauvegardée dans: images_problematiques.txt")
        else:
            print("\nToutes les images ont des annotations valides!")

    except Exception as e:
        print(f"Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())