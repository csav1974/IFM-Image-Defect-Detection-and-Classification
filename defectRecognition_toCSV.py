import cv2
import numpy as np
import tensorflow as tf
import os
from csv_handling import write_defect_data

def defect_recognition(image_path = None, model_name = None):

    # Load Model. Assuming model is already trained
    path_to_model = os.path.join("kerasModels", model_name)
    model = tf.keras.models.load_model(f"{path_to_model}.keras")
    IMG_SIZE = 32  # scales the patch size down (or up) to 32*32 Pixel

    # Load the microscope image
    image = cv2.imread(image_path)
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # Define patch size
    patch_size = 120

    # Define stride (optional)
    stride = patch_size // 2  # 50% overlap if stride = patch_size // 2

    start_x = 0
    start_y = 0
    square_size = 0

    # this part is used if only a part of the image should be examed for faster runtime.
    # Comment out if you want to exam the whole picture
    # ############

    # height, width = work_image.shape
    # # Define the size of the square
    # square_size = 2000

    # # Calculate the top-left corner of the square
    # start_x = (width - square_size) // 5
    # start_y = (height - square_size) // 5
    # # Crop the square around the center
    # image = image[start_y:start_y + square_size, start_x:start_x + square_size]
    # work_image = work_image[start_y:start_y + square_size, start_x:start_x + square_size]

    # ###########


    height, width = work_image.shape

    number_of_patches = len(range(0, height - patch_size, stride)) * len(
        range(0, width - patch_size, stride)
    )
    current_patch_number = 0

    data_list = []

    # Divide the image into patches
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            patch = work_image[y : y + patch_size, x : x + patch_size]
            patch_resized = cv2.resize(
                patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )
            patch_resized = np.expand_dims(
                patch_resized / 255.0, axis=0
            )  # Normalize and add batch dimension

            # ignore black area around probe
            if np.any(patch_resized == 0):
                current_patch_number += 1
                continue
            # Classify the patch
            prediction = model.predict(patch_resized)

            row = [x, y, prediction[0]]
            data_list.append(row)

            current_patch_number += 1
            print(f"Patch: {current_patch_number} / {number_of_patches}")
            # print(prediction)

    write_defect_data(filename=image_path, patch_size=patch_size, stride=stride, data_list=data_list)


defect_recognition(image_path="sampleOnlyBMP/20240424_A2-2m$3D_10x.bmp", model_name="fullModel_v2")