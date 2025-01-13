import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from enumDefectTypes import DefectType

def save_polygons_to_bmp(image, merged_polygons):
    """
    Extracts the image sections enclosed by polygons and saves them as .bmp files.

    Args:
        image (numpy.ndarray): The original image from which to extract the polygons.
        merged_polygons (list): A list of dictionaries containing 'polygon', 'color', and 'defect_type'.
        output_directory (str): Directory where the output images will be saved.
    """
    # Assuming the script is always run from the project root directory (or a known directory within it)
    project_root = os.getcwd()

    # Specify the target folder within the project root
    output_directory = os.path.join(project_root, "dataCollection/Data/testPolygons")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Counter to ensure unique filenames
    counter = 0

    for item in merged_polygons:
        polygon = item['polygon']
        color = item['color']
        defect_type = DefectType(item['defect_type'])

        # Get the bounding box of the polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Convert bounds to integer pixel values
        minx = int(minx)
        miny = int(miny)
        maxx = int(maxx)
        maxy = int(maxy)

        # Create a mask for the polygon
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Get the coordinates of the polygon as a NumPy array
        exterior_coords = np.array(polygon.exterior.coords).round().astype(np.int32)

        # Fill the polygon on the mask
        cv2.fillPoly(mask, [exterior_coords], 255)

        # Extract the ROI using the mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Crop the image to the bounding box of the polygon
        cropped_image = masked_image[miny:maxy, minx:maxx]


        defect_directory = os.path.join(output_directory, defect_type.value)
        if not os.path.exists(defect_directory):
            os.makedirs(defect_directory)
        # Generate a unique filename
        filename = f"polygon_{counter}_defect_{defect_type.value}.bmp"
        filepath = os.path.join(defect_directory, filename)

        # Save the cropped image
        cv2.imwrite(filepath, cropped_image)

        print(f"Saved polygon image to {filepath}")

        counter += 1