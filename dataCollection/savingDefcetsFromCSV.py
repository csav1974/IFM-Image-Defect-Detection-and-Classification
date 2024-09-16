import cv2
import os
import csv
from enumDefectTypes import DefectType


def read_from_csv(csv_path):

    coordinates = []

    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader)    # skip first header line (start_x,start_y,image_size,patch_size,image_name)
        first_row = next(reader)  
        start_x, start_y, image_size, patch_size, defect_type, filename = first_row
        start_x, start_y, image_size, patch_size = int(start_x), int(start_y), int(image_size), int(patch_size)
        if defect_type == DefectType.CHIPPING.value:
            defect_type = DefectType.CHIPPING
        if defect_type == DefectType.WHISKERS.value:
            defect_type = DefectType.WHISKERS
        image_name = os.path.splitext(filename)[0]
        next(reader)    # to skip values  
        next(reader)    # to skip second header (x,y)
        for row in reader:
            coordinates.append((int(row[0]), int(row[1]), int(row[2]))) # reads x, y, patchsize
    
    return start_x, start_y, image_size, patch_size, defect_type, image_name, coordinates

def safeDefectsFromCSV(image_path, csv_path):

    image = cv2.imread(image_path)
    start_x, start_y, image_size, patch_size, defect_type, image_name, coordinates = read_from_csv(csv_path)
    image_name = os.path.splitext(os.path.split(image_name)[-1])[0] # Extract the last folder name
    if image_size == 0:
        image_resized = image
    else :
        image_resized = image[start_y:start_y + image_size, start_x:start_x + image_size]
    rois = []
    for (x, y, patch_size) in coordinates:
        roi = image_resized[y : y + patch_size, x : x + patch_size]
        rois.append(roi)

    base_path = "dataCollection/detectedErrors/machinefoundErrors"
    final_path = os.path.join(base_path, image_name)
    # Creates the subfolder if it doesn’t already exist
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    defect_subfolder = os.path.join(final_path, defect_type.value)
    if not os.path.exists(defect_subfolder):
        os.makedirs(defect_subfolder)
    # Save each ROI as a BMP file
    for idx, roi in enumerate(rois):
        filename = os.path.join(final_path, defect_type.value, f"{defect_type.value}_{idx + 1}.bmp")
        cv2.imwrite(filename, roi)
        print(f"files safed to {filename}")

    

image_path = "sampleOnlyBMP/20240610_A6-2m_10x$3D_Square.bmp"
csv_path = "defectPositionCSV/Chipping/20240610_A6-2m_10x$3D_Square.csv"



safeDefectsFromCSV(image_path, csv_path)

