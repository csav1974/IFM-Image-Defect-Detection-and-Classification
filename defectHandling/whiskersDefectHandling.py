import numpy as np
from defectHandling.calculate_mean_background import images_to_mean_noise as mean_noise
import cv2





def calculate_defect_map_whiskers(coordinates, image, threshold = 0.05):

    background_value = mean_noise("dataCollection/detectedErrors/machinefoundErrors/20240610_A6-2m_10x$3D/No_Error") / 255
    defect_map = np.ones_like(np.array(image)[..., 0])
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ksize = (21, 21)  # Kernelgröße
    sigmaX = 1.0  # Standardabweichung in X-Richtung
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX) / 255

    
    for (x, y, patch_size) in coordinates:
        # Extrahiere den Patch
        patch = blurred_image[y : y + patch_size, x : x + patch_size]
        
        # Berechne die Summe der absoluten Abweichungen der BGR-Werte von den Hintergrundwerten
        deviation_sum = np.sum(np.abs(patch - background_value), axis=-1)

        # Erstelle eine Maske, wo die Abweichung die Schwelle überschreitet
        mask = deviation_sum > threshold

        mask = np.transpose(mask)
        # Set corresponding pixels in defect_map to 0 where the mask is True
        defect_map[x:x + patch_size, y:y + patch_size][mask] = 0
    
    print(np.sum(defect_map == 0))  # Anzahl der Defekte ausgeben
    return defect_map







# def calculate_defect_map_whiskers(coordinates, image, threshold = 0.05):

#     background_value = mean_noise("dataCollection/detectedErrors/machinefoundErrors/20240610_A6-2m_10x$3D/No_Error") / 255
#     defect_map = np.ones_like(np.array(image)[..., 0])
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
#     ksize = (13, 13)  # Kernelgröße
#     sigmaX = 1.0  # Standardabweichung in X-Richtung
#     blurred_image = cv2.GaussianBlur(image_gray, ksize, sigmaX) / 255

    
#     for (x, y, patch_size) in coordinates:
#         # Extrahiere den Patch
#         patch = blurred_image[y : y + patch_size, x : x + patch_size]
        
#         mask = abs(patch - background_value) > threshold
#         mask = np.transpose(mask)
#         # Set corresponding pixels in defect_map to 0 where the mask is True
#         defect_map[x:x + patch_size, y:y + patch_size][mask] = 0
    
#     print(np.sum(defect_map == 0))  # Anzahl der Defekte ausgeben
#     return defect_map