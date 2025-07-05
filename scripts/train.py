import tensorflow as tf
from tf import keras
import os


from scripts.model import compute_base_model, prepare_model_for_tf
from scripts.dataset import prepared_dataset, prepared_dataset_for_tf

device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def train_model():
    # MobileNet
    model = compute_base_model()

    # Load the datasets
    train_ds, test_ds, val_ds = prepared_dataset()

    # Fit the model
    model.fit(train_ds, epochs=50, validation_data=val_ds,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])

    # Save the model
    return model


def transfer_learning_model(model: keras.Model):
    # Prepare the model for transfer learning
    model = prepare_model_for_tf(model)

    # Load the real dataset for transfer learning
    real_train_ds, real_val_ds = prepared_dataset_for_tf()

    # Fine Tuning
    print("Début du fine-tuning...")

    history = model.fit(
        real_train_ds,
        epochs=60,
        validation_data=real_val_ds,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
        ]
    )

    print("Modèle fine-tuné sauvegardé sous 'mobilenet_finetuned.keras'")

    return model, history, real_val_ds


def complete_training():
    """
    Completes the training process by training the model and applying transfer learning.
    """
    # Train the model
    model = train_model()

    # Transfer learning
    return transfer_learning_model(model)
