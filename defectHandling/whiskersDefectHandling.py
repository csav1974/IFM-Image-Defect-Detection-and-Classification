import numpy as np
from defectHandling.calculate_mean_background import images_to_mean_noise as mean_noise
import cv2

def calculate_defect_map_whiskers(coordinates, image, threshold=0.05, patch_size=32, background_value=None):
    if background_value is None:
        background_value = (
            mean_noise(
                "dataCollection/trainingData_20240424_A2-2m$3D_10x/No_Error", 1000
            )
            / 255
        )

    # Initialize defect map with ones (non-defective areas)
    defect_map = np.ones_like(image[..., 0])

    # Apply Gaussian blur to the image and normalize
    ksize = (5, 5)  # Kernel size
    sigmaX = 1.0    # Standard deviation in X direction
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX) / 255.0

    for x, y, _ in coordinates:
        # Extract the patch from the blurred image
        patch = blurred_image[y : y + patch_size, x : x + patch_size]
        height, width, _ = patch.shape

        # Initialize patch mask with False (no defects)
        patch_mask = np.zeros((height, width), dtype=bool)

        # Loop over the patch in blocks of size 5x5 with a stride of 3
        for i in range(0, height - 4, 3):
            for j in range(0, width - 4, 3):
                block = patch[i : i + 5, j : j + 5, :]  # Extract 5x5 block
                block_mean = np.mean(block, axis=(0, 1))  # Compute mean color

                # Calculate deviation from the background value
                deviation = np.sum(np.abs(block_mean - background_value))

                # If deviation exceeds threshold, mark as defect
                if deviation > threshold:
                    patch_mask[i : i + 5, j : j + 5] = True

        # Transpose patch_mask to align axes correctly
        # patch_mask = np.transpose(patch_mask)

        # Update defect_map with detected defects in the patch
        defect_map[y : y + patch_size, x : x + patch_size][patch_mask] = 0

    # After processing, fill enclosed areas in defect_map with black pixels (zeros)
    # Invert defect_map: defects become 1, background becomes 0
    inverted_defect_map = (1 - defect_map).astype(np.uint8)

    # Create a mask for floodFill (dimensions need to be 2 pixels larger than image)
    h, w = inverted_defect_map.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from the borders (assumed to be background)
    # This will fill all background areas connected to the border
    cv2.floodFill(inverted_defect_map, mask, (0, 0), 255)

    # Invert the flood-filled image
    im_floodfill_inv = cv2.bitwise_not(inverted_defect_map)

    # The non-zero regions in im_floodfill_inv are the enclosed areas (holes)
    # Set these enclosed areas to zero in the original defect_map
    defect_map[im_floodfill_inv == 255] = 0

    return defect_map

##########
# This part of the Code is currently not used but can be used to calculate the area of all 
# irregular regions on the sample. However, it only gives a routh hint for the defect area.
##########
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
