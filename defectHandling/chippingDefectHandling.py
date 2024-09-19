import numpy as np
import cv2


def calculate_defect_map_chipping(coordinates, image, threshold=0.9):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

    # Initialize defect_map with ones (same shape as the grayscale image)
    defect_map = np.ones_like(np.array(image)[..., 0])

    # Loop over coordinates and patch sizes
    for x, y, patch_size in coordinates:
        # Extract the patch from the grayscale image
        patch = image_gray[y : y + patch_size, x : x + patch_size]

        # Create a boolean mask for pixels in the patch that exceed the threshold
        mask = patch > threshold
        mask = np.transpose(mask)
        # Apply the mask directly to the defect_map
        defect_map[x : x + patch_size, y : y + patch_size][mask] = 0

    # include_unfound_chipping(defect_map, image)

    # Output the number of defect pixels
    print(f"Defect Pixels from Chipping List: {np.sum(defect_map == 0)}")

    return defect_map


def include_unfound_chipping(defect_map, image, threshold=0.99):
    # Calculate the sum of absolute deviations of the BGR values from the background values
    brightness = np.sum(np.abs(image), axis=-1) / 255
    # Create a mask where the deviation exceeds the threshold
    mask = (brightness > threshold) & (np.sum(image, axis=-1) > 0)
    mask = np.transpose(mask)
    # Set corresponding pixels in defect_map to 0 where the mask is True
    defect_map[mask] = 0

    # Output the number of defect pixels
    print(f"Defect Pixels from all Chipping Spots: {np.sum(defect_map == 0)}")
    return defect_map
