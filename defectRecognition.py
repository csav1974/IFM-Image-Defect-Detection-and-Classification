import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from defectHandling.chippingDefectHandling import create_ones_array_from_image as chipping_handling
from defectHandling.whiskersDefectHandling import create_ones_array_from_image as whiskers_handling
from enumDefectTypes import DefectType


# Load Model. Assuming model is already trained
model_name = "firstModelTest" 
path_to_model = os.path.join("kerasModels", model_name)
model = tf.keras.models.load_model(f"{path_to_model}.keras")
IMG_SIZE = 32  # scales the patch size down (or up) to 32*32 Pixel

# Load the microscope image
filename = 'sampleOnlyBMP/20240610_A6-2m_10x$3D.bmp'
image = cv2.imread(filename)
work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# this part is used if only a part of the image should be examed for faster runtime. 
# Comment out if you want to exam the whole picture  
############

# height, width = work_image.shape
# # Define the size of the square
# square_size = 3000

# # Calculate the top-left corner of the square
# start_x = (width - square_size) // 2 
# start_y = (height - square_size) // 2 
# # Crop the square around the center
# image = image[start_y:start_y + square_size, start_x:start_x + square_size]
# work_image = work_image[start_y:start_y + square_size, start_x:start_x + square_size]

# ###########


height, width = work_image.shape

# Define patch size
patch_size = 64

# Define stride (optional)
stride = patch_size // 2  # 50% overlap if stride = patch_size // 2



number_of_patches = len(range(0, height - patch_size, stride))* len(range(0, width - patch_size, stride))
current_patch_number = 0

# List to store defect positions
defect_positions = []
defect_positions_chipping = []
defect_positions_whiskers = []
chipping_count = 0
whiskers_count = 0

# set the confident threshold of model prediction
chipping_detection_threshold = 0.30
whiskers_detection_threshold = 0.90

# Divide the image into patches
for y in range(0, height - patch_size, stride):
    for x in range(0, width - patch_size, stride):


        patch = work_image[y:y + patch_size, x:x + patch_size]
        patch_resized = cv2.resize(patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        patch_resized = np.expand_dims(patch_resized / 255.0, axis=0)  # Normalize and add batch dimension
        
        # ignore black area around probe
        if np.mean(patch) < 1:
            current_patch_number += 1
            continue
        # Classify the patch
        prediction = model.predict(patch_resized)
        
        if prediction[0][0] > whiskers_detection_threshold or prediction[0][1] > chipping_detection_threshold :  # Assuming threshold for defect = 0.3
            defect_positions.append((x, y))
            if prediction[0][0] > whiskers_detection_threshold :
                whiskers_count += 1
                defect_positions_whiskers.append((x,y))
            if prediction[0][1] > chipping_detection_threshold :
                chipping_count += 1
                defect_positions_chipping.append((x, y))

        current_patch_number += 1
        print(f"Patch: {current_patch_number} / {number_of_patches}")
        # print(f"Preddiction= {prediction}")
        # print(f"Chipping prediction: {prediction[0]} \nWhiskers prediction: {prediction[1]}")


# this part is for data collection
# only used for training the model further
########

# image_for_rois = image
# roisave_chipping = []
# roisave_whiskers = []
# for (x, y) in defect_positions_chipping:

#     roisave_chipping.append(image_for_rois[y : y + patch_size, x : x + patch_size])

# for (x, y) in defect_positions_whiskers:

#     roisave_whiskers.append(image_for_rois[y : y + patch_size, x : x + patch_size])


# def saferois_chipping(rois_save):
#     filemanagement.saveROIsToBMP(rois=rois_save, defectType=filemanagement.DefectType.CHIPPING, subfolder_name="trainingdata/machinefoundErrors")


# def saferois_whiskers(rois_save):
#     filemanagement.saveROIsToBMP(rois=rois_save, defectType=filemanagement.DefectType.WHISKERS, subfolder_name="trainingdata/machinefoundErrors")

# saferois_chipping(rois_save=roisave_chipping)
# saferois_whiskers(rois_save=roisave_whiskers)

############



def calculate_defect_area():
    defect_array_chipping = chipping_handling(image= image, chipping_position_list= defect_positions_chipping, patch_size= patch_size, threshold= 150)
    defect_array_whiskers = whiskers_handling(image= image, whiskers_position_list= defect_positions_whiskers, patch_size= patch_size, threshold= 200)

    # arrays = [defect_array_chipping, defect_array_whiskers]
    arrays = [defect_array_chipping]
    stacked_arrays = np.stack(arrays, axis=0)
    combined_array = np.all(stacked_arrays, axis=0).astype(int)

    num_zeros = np.sum(combined_array == 0)
    num_ones = np.sum(combined_array == 1)
    
    # calculate defect area
    if num_ones == 0:
        ratio = float('inf')  
    else:
        ratio = (num_zeros / num_ones) * 100
    
    # print results
    print(f"\nNumer of defect Pixels: {num_zeros}")
    print(f"Number of working Pixels: {num_ones}")
    print(f"Ratio of defect to working: {ratio:.2f}%")
    save_coordinates_to_csv(defect_positions_chipping, DefectType.CHIPPING)
    save_coordinates_to_csv(defect_positions_whiskers, DefectType.WHISKERS)



def save_coordinates_to_csv(coordinates, defect_type : DefectType, csv_filename="coordinates.csv",):
    folder_name = "defectPositionCSV"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    file_path = os.path.join(folder_name, defect_type.value, csv_filename)
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_x", "start_y","image_size", "patch_size", "image_name"])
        if 'start_x' not in globals:
            start_x = 0
            start_y = 0
            square_size = 0
        writer.writerow([start_x, start_y, square_size, patch_size, filename])
        writer.writerow(["x", "y"])  # Header 
        writer.writerows(coordinates)  # Koordinates
    
    print(f"CSV-Datei wurde erfolgreich unter {file_path} gespeichert.")




# shows all areas that were marked as defects as squares on RGB Probe-Image
def visual_representation(image, defect_positions_chipping, defect_positions_whiskers):

    for (x, y) in defect_positions_whiskers:
        cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), (0, 0, 0), 1)
    for (x, y) in defect_positions_chipping:
        cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), (0, 0, 255), 1)
    cv2.imshow('Detected Defects', image_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# calculate_defect_area()

# Display the result
print(f"\nNumber of Defects found: {len(defect_positions)}")
print(f"Chipping Defects: {chipping_count}\n" 
      f"Whiskers Defects: {whiskers_count}")

image_to_show = image
visual_representation(image_to_show, defect_positions_chipping=defect_positions_chipping, defect_positions_whiskers=defect_positions_whiskers)
