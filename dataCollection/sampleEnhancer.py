import os
import cv2
import numpy as np


def augment_images(folder_path):
    """
    Augments image data in a folder by applying various transformations to
    increase the dataset size for training models.

    Transformations:
    - Rotation by 90°, 180°, 270°
    - Horizontal and vertical flipping
    - Brightness and contrast adjustment


    Note: This function should not be run multiple times on the same folder
    to avoid duplicate images.
    """

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    # Create output folder for augmented images
    output_folder = os.path.join(folder_path, "augmented")
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")) and os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path)

            # Save the original image
            base_name, ext = os.path.splitext(filename)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_original{ext}"), image)

            # List of transformations
            transformations = []

            # Rotation by 90°, 180°, 270°
            rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
            for i, rot in enumerate(rotations):
                rotated = cv2.rotate(image, rot)
                transformations.append((rotated, f"rotated_{(i+1)*90}"))

            # Horizontal flip
            flipped_h = cv2.flip(image, 1)
            transformations.append((flipped_h, "flipped_horizontal"))

            # Vertical flip
            flipped_v = cv2.flip(image, 0)
            transformations.append((flipped_v, "flipped_vertical"))




            # Brightness and contrast adjustment
            for alpha in [0.9, 1.1]:  # Contrast control
                for beta in [-10, 10]:  # Brightness control
                    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                    transformations.append((adjusted, f"brightness_{beta}_contrast_{alpha}"))

        
            # Save all transformed images
            for img, trans_name in transformations:
                output_filename = f"{base_name}_{trans_name}{ext}"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, img)

    print(f"Image augmentation completed. Transformed images are saved in the folder '{output_folder}'.")


# Example function call
augment_images("dataCollection/Data/Training20240829_v2/Whiskers")
