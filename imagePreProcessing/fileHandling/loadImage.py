import cv2
from PIL import Image
import numpy as np

def load_large_image(image_path):
    """
    used to load large Images. cv2 has a problem with loading images over 1gb. 
    This function uses pillow to pre load images in that case.

    Args:
        image_path (str): Path to input Image
    
    Returns: 
        image (np.ndarray): loaded image as numpy Array
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("cv2 could not read image.")
        return image
    
    except Exception as e:
        print(e)
        
        # Fallback on Pillow, if cv2 fails
        try:
            pil_image = Image.open(image_path)
            image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image_cv2
        except Exception as pil_e:
            print(pil_e)
            return None