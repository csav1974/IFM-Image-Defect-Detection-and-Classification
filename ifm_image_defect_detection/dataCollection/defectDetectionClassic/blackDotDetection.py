import cv2
import numpy as np
import csv
import os
from ifm_image_defect_detection.defectHandling.calculate_mean_background import images_to_mean_noise as mean_noise
from ifm_image_defect_detection.enumDefectTypes import DefectType


def mark_black_dots(image, threshold=0.1):
    background_value = (
        mean_noise(
            "dataCollection/detectedErrors/machinefoundErrors/20240610_A6-2m_10x$3D/No_Error", 1000
        )
        / 255
    )
    defect_map = np.ones_like(np.array(image)[..., 0])
    ksize = (3, 3)  # Kernelgröße
    sigmaX = 1.0  # Standardabweichung in X-Richtung
    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX) / 255

    deviation_sum = np.sum(np.abs(blurred_image - background_value), axis=-1)

    # Erstelle eine Maske, wo die Abweichung die Schwelle überschreitet
    mask = deviation_sum > threshold

    mask = np.transpose(mask)
    # Set corresponding pixels in defect_map to 0 where the mask is True
    defect_map[mask] = 0

    return defect_map


def safe_coordinates_to_CSV(
    coordinates,
    defect_type: DefectType,
    start_x=0,
    start_y=0,
    square_size=0,
    patch_size=0,
    filename="unknown",
):
    csv_filename = os.path.splitext(os.path.split(filename)[-1])[0]
    folder_name = "defectPositionCSV"
    folder_path = os.path.join(folder_name, defect_type.value)
    file_path = os.path.join(folder_path, f"{csv_filename}.csv")

    # Erstellen des Ordners, falls er nicht existiert
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Schreiben der Koordinaten in die CSV-Datei
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Kopfzeile für zusätzliche Informationen
        writer.writerow(
            [
                "start_x",
                "start_y",
                "image_size",
                "patch_size",
                "Defect Type",
                "image_name",
            ]
        )
        writer.writerow(
            [start_x, start_y, square_size, patch_size, defect_type.value, filename]
        )

        # Kopfzeile für die Koordinaten
        writer.writerow(["x", "y", "patch_size"])
        # Koordinaten in die Datei schreiben
        writer.writerows(coordinates)

    print(f"CSV-Datei wurde erfolgreich unter {file_path} gespeichert.")


# Read the microscope image
# Replace 'microscope_image.jpg' with your image file path
image_path = "sampleOnlyBMP/20240610_A6-2m_10x$3D.bmp"

image = cv2.imread(image_path)

height, width, _ = image.shape
# Define the size of the square
square_size = 4000

# Calculate the top-left corner of the square
start_x = (width - square_size) // 3
start_y = (height - square_size) // 3
# Crop the square around the center
image = image[start_y : start_y + square_size, start_x : start_x + square_size]

thresh = mark_black_dots(image) * 255

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to store defect data
defects = []

# Loop over the contours
for cnt in contours:
    # Calculate area of the contour
    area = cv2.contourArea(cnt)
    print(area)
    # Ignore small or large contours that are unlikely to be defects
    if area < 20 or area > 1000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    # Calculate the equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi)

    # Approximate the contour to a circle and compute circularity
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Consider contours that are roughly circular
    if circularity > 0.4:
        # Calculate centroid of the defect
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Add the defect information to the list
        defects.append((x, y, max([w, h])))


image_name = os.path.basename(image_path)  # Extract image name from path
image_size = square_size  # The size of the cropped square

safe_coordinates_to_CSV(defects, DefectType.BLACK_DOTS)

# Optionally, draw the detected defects on the image and display it
for defect in defects:
    cv2.rectangle(
        image,
        (defect[1], defect[0]),
        (defect[1] + defect[2], defect[0] + defect[2]),
        (0, 0, 255),
        2,
    )

# Resize the image
resized_image = cv2.resize(thresh, (800, 800))

# Display the resized image with detected defects
cv2.imshow("Detected Defects", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
