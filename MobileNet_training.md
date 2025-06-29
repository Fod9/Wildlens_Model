# MobileNet training specs

## Goal

The goal is to have a MobileNet that can classify animal footprints with high accuracy, leveraging YOLO for detection and MobileNet for classification.

The MobileNet model should be trained with tensorflow

## Training Steps

For more information, refer to the notebook `notebooks/models/mobilenet.ipynb`.

1. **Prepare Dataset**: Use the YOLO-preprocessed datasets (OAT and real) for training MobileNet.
    - Use the `YoloInference` class to preprocess prepare the preprocessed datasets (OAT and real).
    - Ensure the dataset contains crops of animal footprints, either from YOLO detections or center crops as a fallback.
    - Resize the crops to 224x224 pixels and normalize them for MobileNet input.
2. **Extract OAT Dataset Features**
   - Data augmentation using sequential transformations with keras (e.g., random rotations, flips, brightness adjustments).
   - Load the preprocessed OAT dataset.
   - Apply transfer learning on OAT preprocessed dataset crops to extract features.
3. **Fine-tune MobileNet on the Real Dataset**:
   - Train the MobileNet model on these crops, focusing on the final classification layer first.
   - Unfreeze the last N layers of the MobileNet backbone for fine-tuning with a lower learning rate (see the notebooks/models/mobilenet.ipynb file).
4. **Evaluate Model**: Use the validation set to evaluate the model's performance.
    - Check accuracy, precision, recall, and F1 score.
    - Use confusion matrix to analyze misclassifications.

## Expected Output

The final MobileNet model should achieve high accuracy on the validation set, with a well-balanced precision and recall across species classes. The model should be robust enough to handle variations in footprint appearance and background noise.

## Dataset Structures

### OAT Dataset (`data/OpenAnimalTracks_spokay/cropped_imgs/`)
```
cropped_imgs/
├── train/
│   ├── american_mink/
│   ├── beaver/
│   ├── blackbear/
│   └── ... (18 classes total)
├── test/
└── val/
```
- Already split into train/test/val
- 18 animal classes
- Images are pre-cropped footprints

### Real Dataset (`data/dataset_no_oat_downsample_spokay/`)
```
dataset_no_oat_downsample_spokay/
├── castor/
├── chat/
├── chien/
├── coyotte/
├── ecureuil/
├── lapin/
├── loup/
├── lynx/
└── ... (13 classes total)
```
- No predefined splits (needs 80/20 train/val split)
- 13 animal classes (French names)
- Images need preprocessing

## Technical Specifications

### Model Configuration
- **Architecture**: MobileNetV3Small
- **Input Shape**: (224, 224, 3)
- **Pre-trained Weights**: ImageNet
- **Final Layers**: GlobalAveragePooling2D + Dense(512) + Dropout(0.3) + Dense(classes)

### Training Parameters
- **Stage 1 Optimizer**: Adamax
- **Stage 2 Optimizer**: Adam (lr=1e-5)
- **Batch Size**: 16 for OAT, 32 for Real dataset
- **Early Stopping**: Patience=3 (OAT), Patience=5 (Real)
- **Data Augmentation**: Random flips, rotations, zoom, contrast

### Data Augmentation Pipeline
```python
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
])
```

## Expected Results
- **OAT Training**: ~50 epochs, validation accuracy plateau around 50-60%
- **Real Dataset Fine-tuning**: ~43 epochs, final validation accuracy ~75%
- **Model Size**: ~3-5 MB saved model
- **Training Time**: ~30-45 minutes per stage (with GPU)

## Notes for Claude

- Ensure the MobileNet model is saved in a format compatible with the inference pipeline
- The 2 dataset structures are different - adapt code to handle both
- Use `image_dataset_from_directory()` from Keras for both datasets
- Apply transfer learning best practices: freeze backbone initially, then fine-tune
- Monitor for overfitting and use early stopping appropriately
- Save intermediate models after each training stage