import cv2
import os
from enumDefectTypes import DefectType
from csv_handling import read_from_csv


def safeDefectsFromCSV(image_path, csv_path):
    image = cv2.imread(image_path)
    image_name, patch_size, stride, defect_type, data = (
        read_from_csv(csv_path)
    )
    print(defect_type)
    if defect_type == DefectType.CHIPPING.value:
        defect_type = DefectType.CHIPPING
    if defect_type == DefectType.WHISKERS.value:
        defect_type = DefectType.WHISKERS
    if defect_type == DefectType.SCRATCHES.value:
        defect_type = DefectType.SCRATCHES
    if defect_type == DefectType.NO_ERROR.value:
        defect_type = DefectType.NO_ERROR
    if defect_type == "None":
        defect_type = None    
    rois = []
    for x, y in data:
        roi = image[y : y + patch_size, x : x + patch_size]
        rois.append(roi)

    base_path = "dataCollection/Data/detectedErrors/machinefoundErrors"
    final_path = os.path.join(base_path, image_name)
    # Creates the subfolder if it doesn’t already exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    if defect_type:
        # defect_subfolder = os.path.join(final_path, defect_type.value)
        defect_subfolder = os.path.join(final_path, "allROIS")
    else: 
        defect_subfolder = os.path.join(final_path, "allROIS")
    if not os.path.exists(defect_subfolder):
        os.makedirs(defect_subfolder)
    # Save each ROI as a BMP file
    for idx, roi in enumerate(rois):
        filename = os.path.join(
            final_path, defect_type.value, f"{image_name}_{defect_type.value}_{idx + 1}.bmp"
        )
        cv2.imwrite(filename, roi)
        print(f"files safed to {filename}")




def saveDefectsFromList(image, image_name, data_list, patch_size, defect_type : DefectType):

    rois = []
    for x, y, _ in data_list:
        roi = image[y : y + patch_size, x : x + patch_size]
        rois.append(roi)

    base_path = "dataCollection/Data/detectedErrors/machinefoundErrors"
    final_path = os.path.join(base_path, image_name)
    # Creates the subfolder if it doesn’t already exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    defect_subfolder = os.path.join(final_path, defect_type.value)
    if not os.path.exists(defect_subfolder):
        os.makedirs(defect_subfolder)
    # Save each ROI as a BMP file
    for idx, roi in enumerate(rois):
        filename = os.path.join(
            final_path, defect_type.value, f"{image_name}_{defect_type.value}_{idx + 1}.bmp"
        )
        cv2.imwrite(filename, roi)
        print(f"files safed to {filename}")
