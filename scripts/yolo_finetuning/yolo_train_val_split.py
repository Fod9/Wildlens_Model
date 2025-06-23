import argparse
import os
import random
import shutil

import yaml


def create_dataset_structure(dataset_path):
    dirs_to_create = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]

    for dir_path in dirs_to_create:
        full_path = os.path.join(dataset_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")


def get_image_label_pairs(images_folder, labels_folder):
    image_extensions = ['.png', '.jpeg', '.jpg', '.gif', '.bmp', '.tiff']

    pairs = []

    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' does not exist")
        return pairs

    if not os.path.exists(labels_folder):
        print(f"Warning: Labels folder '{labels_folder}' does not exist")
        labels_folder = None

    # Get all image files
    for file in os.listdir(images_folder):
        file_ext = os.path.splitext(file)[1].lower()
        if file_ext in image_extensions:
            image_path = os.path.join(images_folder, file)

            # Look for corresponding label file
            label_path = None
            if labels_folder:
                base_name = os.path.splitext(file)[0]
                potential_label = os.path.join(labels_folder, f"{base_name}.txt")
                if os.path.exists(potential_label):
                    label_path = potential_label

            pairs.append((image_path, label_path))

    return pairs


def split_dataset(pairs, train_ratio=0.8, seed=42):
    random.seed(seed)
    random.shuffle(pairs)

    train_size = int(len(pairs) * train_ratio)
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:]

    return train_pairs, val_pairs


def copy_files(pairs, dataset_path, split_name):
    images_dir = os.path.join(dataset_path, 'images', split_name)
    labels_dir = os.path.join(dataset_path, 'labels', split_name)

    copied_images = 0
    copied_labels = 0

    for image_path, label_path in pairs:
        # Copy image file
        image_filename = os.path.basename(image_path)
        dest_image_path = os.path.join(images_dir, image_filename)
        shutil.copy2(image_path, dest_image_path)
        copied_images += 1

        # Copy label file if it exists
        if label_path and os.path.exists(label_path):
            label_filename = os.path.basename(label_path)
            dest_label_path = os.path.join(labels_dir, label_filename)
            shutil.copy2(label_path, dest_label_path)
            copied_labels += 1

    print(f"{split_name.capitalize()} set: {copied_images} images, {copied_labels} labels")
    return copied_images, copied_labels


def create_data_yaml(dataset_path, class_names, nc=None):
    if nc is None:
        nc = len(class_names)

    data = {
        'path': os.path.abspath(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': nc,
        'names': class_names
    }

    yaml_path = os.path.join(dataset_path, 'data.yaml')

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Created data.yaml at: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val for YOLO training")
    parser.add_argument("-i", "--images_folder", type=str, required=True,
                        help="Path to folder containing images")
    parser.add_argument("-l", "--labels_folder", type=str,
                        help="Path to folder containing YOLO label files (.txt)")
    parser.add_argument("-d", "--dataset_path", type=str, required=True,
                        help="Path where the dataset structure will be created")
    parser.add_argument("-r", "--train_ratio", type=float, default=0.8,
                        help="Ratio for training set (default: 0.8)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("-c", "--class_names", type=str, nargs='+', default=['footprint'],
                        help="List of class names (default: ['footprint'])")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of moving them")

    args = parser.parse_args()

    # Validate arguments
    if not 0 < args.train_ratio < 1:
        print("Error: train_ratio must be between 0 and 1")
        return

    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder '{args.images_folder}' does not exist")
        return

    print(f"Creating dataset structure at: {args.dataset_path}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Random seed: {args.seed}")
    print(f"Class names: {args.class_names}")
    print("-" * 50)

    # Create dataset directory structure
    create_dataset_structure(args.dataset_path)

    # Find image-label pairs
    print("Finding image-label pairs...")
    pairs = get_image_label_pairs(args.images_folder, args.labels_folder)

    if not pairs:
        print("No image files found!")
        return

    print(f"Found {len(pairs)} image-label pairs")

    # Split dataset
    print("Splitting dataset...")
    train_pairs, val_pairs = split_dataset(pairs, args.train_ratio, args.seed)

    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")

    # Copy files to respective directories
    print("Copying files...")
    copy_files(train_pairs, args.dataset_path, 'train')
    copy_files(val_pairs, args.dataset_path, 'val')

    # Create data.yaml file
    print("Creating data.yaml...")
    create_data_yaml(args.dataset_path, args.class_names)

    print("-" * 50)
    print("Dataset split completed successfully!")
    print(f"Dataset ready for YOLO training at: {args.dataset_path}")


if __name__ == "__main__":
    main()