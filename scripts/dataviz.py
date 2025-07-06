from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

def compute_plot(history: Any):

    # Évaluation du modèle fine-tuné
    
    # Afficher les courbes d'apprentissage
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Perte du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Précision du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    
    plt.tight_layout()
    # plt.show()
    os.makedirs("metrics", exist_ok=True)
    plt.savefig("metrics/mobilenet_finetuned_learning_curves.png")
    

def evaluate_model(model: Any, real_val_ds: Any):
    
    # Évaluer sur l'ensemble de validation
    val_loss, val_accuracy = model.evaluate(real_val_ds)
    print(f"Précision finale sur les données de validation: {val_accuracy:.4f}")
    print(f"Perte finale sur les données de validation: {val_loss:.4f}")
    
    # Matrice de confusion sur les 13 classes
    
    # Obtenir les noms des classes depuis le répertoire de données
    data_dir = "/home/shared/Wildlens/full_dataset_wildlens/dataset_no_oat_downsample"
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Classes: {class_names}")
    print(f"Nombre de classes: {len(class_names)}")
    
    # Générer les prédictions et les vraies étiquettes
    y_true = []
    y_pred = []
    
    print("Génération des prédictions...")
    for batch_idx, (x, y) in enumerate(real_val_ds):
        print(f"Traitement du batch {batch_idx + 1}")
        predictions = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y.numpy())
    
    print(f"Total d'échantillons traités: {len(y_true)}")
    
    # Évaluer le modèle
    val_loss, val_accuracy = model.evaluate(real_val_ds, verbose=0)
    print(f"Précision finale sur les données de validation: {val_accuracy:.4f}")
    print(f"Perte finale sur les données de validation: {val_loss:.4f}")
    
    # Créer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # Afficher la matrice de confusion avec une belle visualisation
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion - Modèle MobileNet (13 Classes)', fontsize=16, pad=20)
    plt.xlabel('Prédictions', fontsize=12)
    plt.ylabel('Vraies Étiquettes', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    #plt.show()

    os.makedirs("metrics", exist_ok=True)
    plt.savefig("metrics/mobilenet_confusion_matrix.png")
    
    # Calculer et afficher les métriques par classe
    print("\n" + "=" * 60)
    print("RAPPORT DE CLASSIFICATION DÉTAILLÉ")
    print("=" * 60)

    print(sorted(set(y_true)))
    print(sorted(set(y_pred)))
    print(len(class_names))

    valid_indices = [i for i, pred in enumerate(y_pred) if pred <= 12]

    # Appliquer le filtre
    y_true = np.array(y_true)[valid_indices]
    y_pred = np.array(y_pred)[valid_indices]

    print(f"Nombre d'échantillons après filtrage: {len(y_true)}")
    print(f"Classes uniques après filtrage: {sorted(set(y_true))}")
    print(f"Classes uniques prédites après filtrage: {sorted(set(y_pred))}")
    print(f"Nombre de classes après filtrage: {len(class_names)}")

    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculer l'accuracy globale
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nPrécision globale: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    # Afficher les erreurs de classification les plus fréquentes
    print(f"\n" + "=" * 40)
    print("ERREURS DE CLASSIFICATION LES PLUS FRÉQUENTES")
    print("=" * 40)
    
    # Enlever la diagonale (prédictions correctes)
    cm_errors = cm.copy()
    np.fill_diagonal(cm_errors, 0)
    
    # Trouver les 5 erreurs les plus fréquentes
    flat_indices = np.argsort(cm_errors.ravel())[-5:]
    row_indices, col_indices = np.unravel_index(flat_indices, cm_errors.shape)
    
    for i in range(len(flat_indices) - 1, -1, -1):
        if cm_errors[row_indices[i], col_indices[i]] > 0:
            true_class = class_names[row_indices[i]]
            pred_class = class_names[col_indices[i]]
            count = cm_errors[row_indices[i], col_indices[i]]
            print(f"{true_class} → {pred_class}: {count} erreurs")
    
    # Afficher les performances par classe
    print(f"\n" + "=" * 40)
    print("PERFORMANCES PAR CLASSE")
    print("=" * 40)
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]  # True positives
        fp = np.sum(cm[:, i]) - tp  # False positives
        fn = np.sum(cm[i, :]) - tp  # False negatives
    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        print(f"{class_name:20s} - Précision: {precision:.3f}, Rappel: {recall:.3f}, F1: {f1:.3f}")
    
    val_loss, val_accuracy = model.evaluate(real_val_ds)
    print(f"Précision finale sur les données de validation: {val_accuracy:.4f}")
    print(f"Perte finale sur les données de validation: {val_loss:.4f}")

    metrics = {
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def vizualize(model: Any, real_val_ds: Any):
    """
    Visualizes the model's performance and saves the results.

    Args:
        model (Any): The trained model.
        real_val_ds (Any): The validation dataset.
    """

    # Compute and plot learning curves
    compute_plot(model.history)

    # Evaluate the model
    return evaluate_model(model, real_val_ds)
