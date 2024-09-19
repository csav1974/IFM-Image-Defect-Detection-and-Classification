import numpy as np
from defectHandling.calculate_mean_background import images_to_mean_noise as mean_noise
import cv2


def calculate_defect_map_whiskers(coordinates, image, threshold=0.05):
    background_value = (
        mean_noise(
            "dataCollection/detectedErrors/machinefoundErrors/20240610_A6-2m_10x$3D/No_Error", 1000
        )
        / 255
    )
    defect_map = np.ones_like(np.array(image)[..., 0])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = (21, 21)  # Kernel size
    sigmaX = 1.0  # Standard deviation in X direction
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX) / 255

    for x, y, patch_size in coordinates:
        patch = blurred_image[y : y + patch_size, x : x + patch_size]

        # Calculate the sum of absolute deviations of the BGR values from the background values
        deviation_sum = np.sum(np.abs(patch - background_value), axis=-1)

        # Create a mask where the deviation exceeds the threshold
        mask = deviation_sum > threshold

        mask = np.transpose(mask)
        # Set corresponding pixels in defect_map to 0 where the mask is True
        defect_map[x : x + patch_size, y : y + patch_size][mask] = 0

    print(
        f"Defect Pixels from Whiskers List: {np.sum(defect_map == 0)}"
    )  # Output the number of detected defects
    return defect_map


def calculate_unknown_defect_area(image, threshold=0.3):
    background_value = (
        mean_noise(
            "dataCollection/detectedErrors/machinefoundErrors/20240610_A6-2m_10x$3D/No_Error"
        )
        / 255
    )
    defect_map = np.ones_like(np.array(image)[..., 0])
    ksize = (5, 5)  # Kernel size
    sigmaX = 1.0  # Standard deviation in X direction
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX) / 255

    # Calculate the sum of absolute deviations of the BGR values from the background values
    deviation_sum = np.sum(np.abs(blurred_image - background_value), axis=-1)

    # Create a mask where the deviation exceeds the threshold
    mask = (deviation_sum > threshold) & (np.sum(image, axis=-1) > 0)

    mask = np.transpose(mask)
    # Set corresponding pixels in defect_map to 0 where the mask is True
    defect_map[mask] = 0

    print(np.sum(defect_map == 0))  # Output the number of detected defects
    return defect_map
