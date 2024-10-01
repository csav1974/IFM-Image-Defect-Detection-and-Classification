import numpy as np
import os
import cv2


def calculate_noise(image):
    ksize = (15, 15)  # Kernelgröße
    sigmaX = 1.0  # Standardabweichung in X-Richtung
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX)
    mean = np.mean(blurred_image, axis=(0, 1))

    return mean

def list_to_mean_noise(image, no_defect_list, patch_size):
    mean_array = []
    for x, y, _ in no_defect_list:
        patch = image[y : y + patch_size, x : x + patch_size]
        mean = calculate_noise(patch)
        mean_array.append(mean)
    return np.mean(mean_array)

def images_to_mean_noise(folderpath, number_of_patches):
    mean_array = []
    image_paths = find_bmp_files(folderpath, number_of_patches)
    for path in image_paths:
        image = cv2.imread(path)
        mean = calculate_noise(image=image)
        mean_array.append(mean)
    
    return np.mean(mean_array)


def find_bmp_files(directory, number_of_patches):
    bmp_files = []

    # Iterate over all files in the specified directory
    for file in os.listdir(directory):
        # Check if the file has a .csv extension
        if file.endswith(".bmp"):
            # Create the full path of the file
            full_path = os.path.join(directory, file)
            # Add the path to the list
            bmp_files.append(full_path)
            if len(bmp_files) > number_of_patches:
                return bmp_files

    return bmp_files
