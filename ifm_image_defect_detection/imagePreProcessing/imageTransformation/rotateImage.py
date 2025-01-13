import cv2
import numpy as np


def rotate_image(image, angle):
    """
    Rotates an image (angle in degrees) and expands the image to avoid cropping.

    Args:
        image (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees.

    Returns:
        rotated (numpy.ndarray): Rotated image.
        M (numpy.ndarray): Rotation matrix used for the transformation.
    """
    # Get image dimensions
    h, w = image.shape[:2]
    # Calculate the center of the image
    cX, cY = w // 2, h // 2

    # Get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

    # Compute sine and cosine of rotation matrix
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute new bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for the translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (nW, nH))

    return rotated, M