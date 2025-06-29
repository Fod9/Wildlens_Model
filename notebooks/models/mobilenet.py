#%%
import tensorflow as tf
import keras

device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"

# Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load data
#%%
train_ds = keras.preprocessing.image_dataset_from_directory(
    "../../data/OpenAnimalTracks_spokay/cropped_imgs/train",
    image_size=(224, 224),
    batch_size=16,
)
test_ds = keras.preprocessing.image_dataset_from_directory(
    "../../data/OpenAnimalTracks_spokay/cropped_imgs/test",
    image_size=(224, 224),
    batch_size=16,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    "../../data/OpenAnimalTracks_spokay/cropped_imgs/val",
    image_size=(224, 224),
    batch_size=16,
)
#%%
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
#%%

model.compile(optimizer="adamax", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])

# Save the model
model.save("mobilenet_oat.keras")
#%%
mobilnet = keras.models.load_model("mobilenet_oat.keras")
mobilenet.trainable = True

fine_tune_at = len(mobilenet.layers) - 20

for layer in mobilenet.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
#%%
model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)])
#%%
real_ds = keras.preprocessing.image_dataset_from_directory(
    "../../data/dataset_no_oat_downsample_spokay",
    image_size=(224, 224),
    batch_size=32,
)
#%%

train_size = int(0.8 * len(real_ds))
val_size = len(real_ds) - train_size

real_train_ds = real_ds.take(train_size)
real_val_ds = real_ds.skip(train_size)

print(f"Dataset d'entraînement: {train_size} batches")
print(f"Dataset de validation: {val_size} batches")

#%%

print("Début du fine-tuning...")


history = mobilnet.fit(
    real_train_ds,
    epochs=60,
    validation_data=real_val_ds,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)
    ]
)


finetuned_model.save("mobilenet_finetuned_with_no_oat.keras")
print("Modèle fine-tuné sauvegardé sous 'mobilenet_finetuned.keras'")

#%%
# Évaluation du modèle fine-tuné
import matplotlib.pyplot as plt

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
plt.show()

# Évaluer sur l'ensemble de validation
val_loss, val_accuracy = finetuned_model.evaluate(real_val_ds)
print(f"Précision finale sur les données de validation: {val_accuracy:.4f}")
print(f"Perte finale sur les données de validation: {val_loss:.4f}")

#%%
