import tensorflow as tf
from tensorflow import keras
import os

train_dir = "/home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks/cropped_imgs/train"
test_dir = "/home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks/cropped_imgs/test"
val_dir = "/home/shared/Wildlens/full_dataset_wildlens/OpenAnimalTracks/cropped_imgs/val"

def prepared_dataset():
    """
    Prepares the dataset for training and validation.
    Returns:
        train_ds: Training dataset.
        test_ds: Testing dataset.
        val_ds: Validation dataset.
    """

    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=16,
    )

    test_ds = keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=16,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=16,
    )

    return train_ds, test_ds, val_ds
# Train Data


def prepared_dataset_for_tf():
    """
    Prepares the dataset for transfer learning with TensorFlow.
    Returns:
        test_ds: Testing dataset.
    """

    real_ds = keras.preprocessing.image_dataset_from_directory(
        "/home/shared/Wildlens/full_dataset_wildlens/dataset_no_oat_downsample",
        image_size=(224, 224),
        batch_size=32,
    )

    # Real Dataset
    train_size = int(0.8 * len(real_ds))
    #val_size = len(real_ds) - train_size

    real_train_ds = real_ds.take(train_size)
    real_val_ds = real_ds.skip(train_size)

    return real_train_ds, real_val_ds