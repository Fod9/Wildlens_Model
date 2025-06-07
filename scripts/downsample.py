import argparse
import os
import cv2

args = argparse.ArgumentParser()

args.add_argument("-f", "--folder")
args.add_argument("-d", "--destination")

def find_smallest_folder_len(folder: str):
    min_size = float("inf")
    for subfolder in os.listdir(folder):
        folder_path = os.path.join(folder, subfolder)
        folder_len = len(os.listdir(folder_path))
        if folder_len < min_size:
            min_size = folder_len

    return min_size


def downsample(folder: str, folder_size: int, destination: str):
    for root, dirs, files in os.walk(folder):
        num_files = 0
        for file in files:
            if os.path.splitext(file)[1] not in [".jpg", ".png", ".jpeg"]:
                continue
            
            if num_files >= folder_size:
                break
            
            filename = file.split("/")[-1]
            file_path = os.path.join(root, file)
            dirname = root.split("/")[-1]
            img = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
            num_files += 1 

            dest_folder = os.path.join(destination, dirname)
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            
            dest_path = os.path.join(dest_folder, filename)
            cv2.imwrite(dest_path, img)

            print(f"image saved to {dest_path}")

            
    

if __name__ == "__main__":
    parsed_args = args.parse_args()
    smallest_folder_size = find_smallest_folder_len(parsed_args.folder)
    downsample(parsed_args.folder, smallest_folder_size, parsed_args.destination)

    