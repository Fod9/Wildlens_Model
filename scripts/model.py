import tensorflow as tf
import keras

def compute_base_model():
    """
    Computes the base model using MobileNetV3Small with data augmentation and additional layers.
    :return: keras.Model: The compiled MobileNetV3Small model with additional layers.
    """

    # Load the MobileNetV3Small model
    mobilenet = keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    # Initially freeze the base model
    mobilenet.trainable = False

    # Data augmentation
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ])

    # Build the complete model
    model = keras.Sequential([
        data_augmentation,
        mobilenet,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(18, activation="softmax"),
    ])

    # Compile
    model.compile(optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def prepare_model_for_tf(model: keras.Model) -> keras.Model:
    """ Prepares the MobileNetV3Small model for transfer learning."""

    # Transfer Learning
    model.trainable = True

    fine_tune_at = len(model.layers) - 20

    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

