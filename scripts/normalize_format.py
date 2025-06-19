import os
import argparse
from PIL import Image

images_extension = [".png", ".jpeg", ".jpg", ".gif", ".bmp", ".tiff"]


def change_ext(folder_path: str):
    for root, _, files in os.walk(folder_path):

        nb_files = len(files)

        num_digits = len(str(nb_files))

        counter = 1

        for file in files:

            lower_ext = os.path.splitext(file)[1].lower()

            if lower_ext in images_extension:

                if lower_ext != ".jpg":
                    image = Image.open(os.path.join(root, file))

                    subfolder = os.path.basename(root)

                    new_filename = f"{subfolder}_\
						{str(counter).zfill(num_digits)}.jpg"

                    image.save(os.path.join(root, new_filename))

                    os.remove(os.path.join(root, file))

                    print(f"{file} has been deleted and saved as {new_filename}")

                    counter += 1

        else:

            subfolder = os.path.basename(root)

            new_filename = f"{subfolder}_{str(counter).zfill(num_digits)}.jpg"

            os.rename(os.path.join(root, file), os.path.join(root, new_filename))

            print(f"{file} has been renamed as {new_filename}")

            counter += 1


parser = argparse.ArgumentParser(description="Delete all non-jpg files in a folder and save them as jpg")

parser.add_argument("-f", "--folder_path", type=str, help="Path to the folder containing the files")

args = parser.parse_args()

if __name__ == "__main__":

    print("Deleting all non-jpg files in the folder and saving them as jpg")

    change_ext(args.folder_path)