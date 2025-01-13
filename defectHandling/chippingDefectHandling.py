import numpy as np
import cv2


def calculate_defect_map_chipping(coordinates, image, threshold=0.8, patch_size = 32):

    """
    Generates a defect map highlighting chipping areas in an image.

    This function processes an input image and marks regions as defects based 
    on the provided coordinates, a threshold value, and patch size. Pixels in 
    the patches exceeding the threshold are marked as defects.

    Parameters:
    ----------
    coordinates : list of tuples
        A list of (x, y, _) coordinates indicating the starting points of patches.
    image : numpy.ndarray
        The input image in BGR format.
    threshold : float, optional
        The intensity threshold for marking pixels as defects (default is 0.8).
    patch_size : int, optional
        The size of the square patches to analyze (default is 32).

    Returns:
    -------
    numpy.ndarray
        A binary defect map where 0 represents defects and 1 represents non-defects.
    """

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255

    # Initialize defect_map with ones (same shape as the grayscale image)
    defect_map = np.ones_like(np.array(image)[..., 0])

    # Loop over coordinates and patch sizes
    for x, y, _ in coordinates:
        # Extract the patch from the grayscale image
        patch = image_gray[y : y + patch_size, x : x + patch_size]

        # Create a boolean mask for pixels in the patch that exceed the threshold
        mask = patch > threshold
        # mask = np.transpose(mask)
        # Apply the mask directly to the defect_map
        defect_map[y : y + patch_size, x : x + patch_size][mask] = 0


    # include_unfound_chipping(defect_map, image_gray)

    return defect_map


def include_unfound_chipping(defect_map, image_gray, threshold=0.90):
    brightness = image_gray
    # Create a mask where the brightness exceeds the threshold
    mask = (brightness > threshold) & (image_gray > 0)
    mask = np.transpose(mask)
    # Set corresponding pixels in defect_map to 0 where the mask is True
    defect_map[mask] = 0

    # Output the number of defect pixels
    print(f"Defect Pixels from all Chipping Spots: {np.sum(defect_map == 0)}")
    return defect_map
