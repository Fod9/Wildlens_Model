import os
import random
import tensorflow as tf
import numpy as np
import keras
from PIL import Image

class CustomDataGen(keras.utils.Sequence):
    def __init__(self, data_folder: str, aug_len: dict[str, float], batch_size=32, split="train", train_ratio=0.8, seed=42):
        super().__init__()

        self.data_folder = data_folder
        self.aug_len = aug_len
        self.batch_size = batch_size
        self.data = []

        self.pipeline = tf.keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.01),
            keras.layers.Rescaling(1./255),
            keras.layers.Resizing(224, 224),
            keras.layers.Normalization(
                mean=(0.485, 0.456, 0.406), 
                variance=(0.229, 0.224, 0.225)
            )
        ])

        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.load_data()
        
        self.classes = {label: idx for idx, label in enumerate(set([item["label"] for item in self.data]))}
        print(self.classes)

    
    def load_data(self):
        all_files = []
        for root, _, files in os.walk(self.data_folder):
            ext = os.path.splitext(files[0])[1]
            if ext not in [".jpg", ".jpeg", ".png"]:
                continue
            subfolder = os.path.basename(root)
            for file in files:
                all_files.append({
                    "path": os.path.join(root, file),
                    "label": subfolder
                })
    
        random.seed(self.seed)
        random.shuffle(all_files)
    
        # Ensure all labels are present in both splits
        label_counts = {label: 0 for label in set([item["label"] for item in all_files])}
        for item in all_files:
            label_counts[item["label"]] += 1
    
        train_files = []
        test_files = []
        for label, count in label_counts.items():
            label_files = [item for item in all_files if item["label"] == label]
            split_idx = int(count * self.train_ratio)
            train_files.extend(label_files[:split_idx])
            test_files.extend(label_files[split_idx:])
    
        if self.split == "train":
            base_files = train_files
        else:
            base_files = test_files
    
        self.data = []
        for file in base_files:
            if self.split == "train" and file["label"] in self.aug_len:
                prob = self.aug_len[file["label"]]
                for _ in range(int(prob)):
                    self.data.append(file)
                if np.random.rand() < (prob - int(prob)):
                    self.data.append(file)
            else:
                self.data.append(file)
    
        random.shuffle(self.data)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for item in batch_data:
            img = Image.open(item["path"])
            img = np.array(img)
            if img.ndim == 2:
                img = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] == 1:
                img = np.concatenate([img] * 3, axis=-1)
            img = self.pipeline(img)
            batch_images.append(img)
            batch_labels.append(self.classes[item["label"]])
        
        batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=len(self.classes))
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        random.shuffle(self.data)