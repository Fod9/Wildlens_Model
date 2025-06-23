import os
import argparse
from PIL import Image

images_extension = [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".tiff"]


def find_label_file(image_filename, labels_folder):
    if not labels_folder or not os.path.exists(labels_folder):
        return None

    # Get base name without extension
    base_name = os.path.splitext(image_filename)[0]
    label_file = os.path.join(labels_folder, f"{base_name}.txt")

    if os.path.exists(label_file):
        return label_file
    return None


def change_ext(folder_path: str, custom_prefix: str = None, labels_folder: str = None):
    for root, _, files in os.walk(folder_path):
        # Filter only image files
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in images_extension]

        if not image_files:
            continue

        nb_files = len(image_files)
        num_digits = len(str(nb_files))
        counter = 1

        # Determine prefix to use
        if custom_prefix:
            prefix = custom_prefix
        else:
            subfolder = os.path.basename(root)
            # Handle case where basename returns empty string (root directory)
            if not subfolder or subfolder == ".":
                prefix = os.path.basename(os.path.abspath(root))
                if not prefix:
                    prefix = "image"
            else:
                prefix = subfolder

        print(f"Processing folder: {root}")
        print(f"Using prefix: {prefix}")
        if labels_folder:
            print(f"Labels folder: {labels_folder}")

        for file in image_files:
            lower_ext = os.path.splitext(file)[1].lower()
            new_base_name = f"{prefix}_{str(counter).zfill(num_digits)}"
            new_image_filename = f"{new_base_name}.jpg"

            # Find corresponding label file
            old_label_file = find_label_file(file, labels_folder)

            if lower_ext != ".jpg":
                try:
                    image = Image.open(os.path.join(root, file))

                    # Save as JPG
                    image.save(os.path.join(root, new_image_filename), "JPEG")

                    # Remove original image file
                    os.remove(os.path.join(root, file))

                    print(f"{file} has been converted and saved as {new_image_filename}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")
                    counter += 1
                    continue

            else:
                # File is already .jpg, just rename it
                # Only rename if the name is different
                if file != new_image_filename:
                    os.rename(os.path.join(root, file), os.path.join(root, new_image_filename))
                    print(f"{file} has been renamed as {new_image_filename}")

            # Handle corresponding label file
            if old_label_file:
                new_label_filename = f"{new_base_name}.txt"
                new_label_path = os.path.join(labels_folder, new_label_filename)

                try:
                    os.rename(old_label_file, new_label_path)
                    print(f"  Label file renamed: {os.path.basename(old_label_file)} -> {new_label_filename}")
                except Exception as e:
                    print(f"  Error renaming label file {old_label_file}: {e}")

            counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all image files to JPG format and normalize filenames")
    parser.add_argument("-f", "--folder_path", type=str, required=True,
                        help="Path to the folder containing the image files")
    parser.add_argument("-p", "--prefix", type=str,
                        help="Custom prefix for filenames (default: folder name)")
    parser.add_argument("-l", "--labels_folder", type=str,
                        help="Path to the folder containing YOLO label files (.txt)")

    args = parser.parse_args()

    if not os.path.exists(args.folder_path):
        print(f"Error: Folder '{args.folder_path}' does not exist")
        exit(1)

    if args.labels_folder and not os.path.exists(args.labels_folder):
        print(f"Warning: Labels folder '{args.labels_folder}' does not exist")
        print("Proceeding without label file renaming...")
        args.labels_folder = None

    print("Converting all non-jpg image files to jpg format...")
    if args.labels_folder:
        print("Also renaming corresponding YOLO label files...")

    change_ext(args.folder_path, args.prefix, args.labels_folder)
    print("Processing complete!")