import numpy as np
import os
import cv2


def calculate_noise(image):
    ksize = (15, 15)  # Kernelgröße
    sigmaX = 1.0  # Standardabweichung in X-Richtung
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX)
    mean = np.mean(blurred_image, axis=(0, 1))

    return mean


def images_to_mean_noise(folderpath):
    mean_array = []
    image_paths = find_bmp_files(folderpath)
    for path in image_paths:
        image = cv2.imread(path)
        mean = calculate_noise(image=image)
        mean_array.append(mean)
    print(f"mean returned:{np.mean(mean_array, axis = 0)}")
    return np.mean(mean_array)


def find_bmp_files(directory):
    bmp_files = []

    # Iterate over all files in the specified directory
    for file in os.listdir(directory):
        # Check if the file has a .csv extension
        if file.endswith(".bmp"):
            # Create the full path of the file
            full_path = os.path.join(directory, file)
            # Add the path to the list
            bmp_files.append(full_path)
            if len(bmp_files) > 500:
                return bmp_files

    return bmp_files
