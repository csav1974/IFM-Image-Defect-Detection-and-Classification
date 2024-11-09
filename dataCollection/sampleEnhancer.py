import os
import cv2
import numpy as np
import random

def augment_images(folder_path, max_new_images):
    """
    Augments image data in a folder by applying various transformations to
    increase the dataset size for training models.

    Transformations:
    - Rotation by 90°, 180°, 270°
    - Horizontal and vertical flipping
    - Brightness and contrast adjustment

    Note: This function should not be run multiple times on the same folder
    to avoid duplicate images.

    Parameters:
    - folder_path: Path to the folder containing images to augment.
    - max_new_images: Maximum number of new images to create.
    """

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    # Create output folder for augmented images
    output_folder = os.path.join(folder_path, "augmented")
    os.makedirs(output_folder, exist_ok=True)

    augmented_images = []  # List to store augmented images (image data and filename)

    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")) and os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path)

            base_name, ext = os.path.splitext(filename)

            # List of transformations
            transformations = []

            # Rotation by 90°, 180°, 270°
            rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
            for i, rot in enumerate(rotations):
                rotated = cv2.rotate(image, rot)
                transformations.append((rotated, f"{base_name}_rotated_{(i+1)*90}{ext}"))

            # Horizontal flip
            flipped_h = cv2.flip(image, 1)
            transformations.append((flipped_h, f"{base_name}_flipped_horizontal{ext}"))

            # Vertical flip
            flipped_v = cv2.flip(image, 0)
            transformations.append((flipped_v, f"{base_name}_flipped_vertical{ext}"))

            # Brightness and contrast adjustment
            for alpha in [0.9, 1.1]:  # Contrast control
                for beta in [-10, 10]:  # Brightness control
                    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                    transformations.append((adjusted, f"{base_name}_brightness_{beta}_contrast_{alpha}{ext}"))

            # Collect all transformed images
            augmented_images.extend(transformations)

    # Check if the number of augmented images exceeds the maximum
    if len(augmented_images) > max_new_images:
        augmented_images = random.sample(augmented_images, max_new_images)

    # Save the selected augmented images
    for img, output_filename in augmented_images:
        output_path = os.path.join(folder_path, output_filename)
        cv2.imwrite(output_path, img)

    print(f"Image augmentation completed. {len(augmented_images)} images are saved in the folder '{output_folder}'.")

# Example function call with max_new_images set to 100
augment_images("dataCollection/Data/Training20240829_v3/No_Error", max_new_images=12000)
