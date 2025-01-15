import cv2
import os
from ifm_image_defect_detection.enumDefectTypes import DefectType


def saveDefectsFromList(image, image_name, data_list, patch_size, defect_type : DefectType):

    # Assuming the script is always run from the project root directory (or a known directory within it)
    project_root = os.getcwd()

    # Specify the target folder within the project root
    base_path = os.path.join(project_root, "dataCollection/Data/detectedErrors/machinefoundErrors")

    rois = []
    for x, y, _ in data_list:
        roi = image[y : y + patch_size, x : x + patch_size]
        rois.append(roi)

    final_path = os.path.join(base_path, image_name, defect_type.value)
    # Creates the subfolder if it doesn’t already exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    # Save each ROI as a BMP file
    for idx, roi in enumerate(rois):
        filename = os.path.join(
            final_path, f"{image_name}_{defect_type.value}_{idx + 1}.bmp"
        )
        cv2.imwrite(filename, roi)
        print(f"files safed to {filename}")