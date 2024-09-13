import os
from enum import Enum
import cv2 as cv

# class for accepted Defect-Types
class DefectType(Enum):
    WHISKERS = "Whiskers"
    CHIPPING = "Chipping"
    NO_ERROR = "No_Error"

def find_largest_file(directory):
    """
    Finds the largest file in the specified directory.
 
    Args:
        directory (str): The path to the directory to search.

    Returns:
        str: The full path to the largest file found, or None if the directory is empty or no files are found.
    """
    largest_file = None
    max_size = 0

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)

            # Update if a larger file is found
            if file_size > max_size:
                max_size = file_size
                largest_file = file_path

    return largest_file


def saveROIsToBMP(rois, defectType: DefectType, subfolder_name, base_folder="dataCollection/detectedErrors"):
    """
    Saves each ROI in rois as a BMP file in a specified subfolder within 'detectedErrors'.

    Args:
        circle_rois (list): A list of ROI images to save.
        subfolder_name (str): The name of the subfolder where the files will be saved.
            This should be the path to the processed File to make sure the new created folder of Errors can be matched with original Image
        base_folder (str, optional): The base folder where the subfolder will be created. Defaults to 'detectedErrors'.

    Returns:
        None
    """
    # Normalize the path to handle any inconsistencies
    normalized_path = os.path.normpath(subfolder_name)
    # Extract the last folder name
    last_folder = os.path.basename(normalized_path)
    # Create the full path to the subfolder
    folder_path = os.path.join(base_folder, last_folder)

    final_path = os.path.join(folder_path, os.path.normpath(defectType.value))

    # Creates the subfolder if it doesn’t already exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    # Save each ROI as a BMP file
    for idx, roi in enumerate(rois):
        filename = os.path.join(final_path, f"{defectType.value}_{idx + 1}.bmp")
        cv.imwrite(filename, roi)

    print(f'files where safed to {final_path}')
