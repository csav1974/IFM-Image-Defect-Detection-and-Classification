import cv2
import numpy as np
import tensorflow as tf
import os
from csvHandling.writePredictionCSV import write_defect_data
from tqdm import tqdm

def defect_recognition(image_path = None, model_name = None):

    # Load Model. Assuming model is already trained
    path_to_model = os.path.join("kerasModels", model_name)
    model = tf.keras.models.load_model(f"{path_to_model}.keras")
    IMG_SIZE = 128  # scales the patch resolution down to IMG_SIZE*IMG_SIZE Pixel

    # Load the microscope image
    image = cv2.imread(image_path)
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define patch size
    patch_size = 128

    # Define stride (optional)
    stride = patch_size  // 2 # 50% overlap if stride = patch_size // 2

    # this part is used if only a part of the image should be examed for faster runtime. 
    # Only used for quick testing.
    # Comment out if you want to exam the whole picture
    # ############

    # start_x = 0
    # start_y = 0
    # square_size = 0

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
            # progress_in_perzent = current_patch_number * 100 / number_of_patches
            # print(f"{progress_in_perzent:.2f}%")
            # print(prediction)

    write_defect_data(filename=image_path, patch_size=patch_size, stride=stride, data_list=data_list)


def defect_recognition_parallel_computing(image_path=None, model_name=None):

    # Load the model (assuming it's already trained)
    path_to_model = os.path.join("kerasModels", model_name)
    model = tf.keras.models.load_model(f"{path_to_model}.keras")
    IMG_SIZE = 128  # Scales the patch resolution to IMG_SIZE x IMG_SIZE pixels

    # Load the microscope image
    image = cv2.imread(image_path)
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # this part is used if only a part of the image should be examed for faster runtime. 
    # Only used for quick testing.
    # Comment out if you want to exam the whole picture
    ############

    # start_x = 0
    # start_y = 0
    # square_size = 0

    # height, width = work_image.shape
    # # Define the size of the square
    # square_size = 800

    # # Calculate the top-left corner of the square
    # start_x = (width - square_size) // 5
    # start_y = (height - square_size) // 5
    # # Crop the square around the center
    # image = image[start_y:start_y + square_size, start_x:start_x + square_size]
    # work_image = work_image[start_y:start_y + square_size, start_x:start_x + square_size]

    # ###########


    # Define patch size
    patch_size = 128

    # Define stride (optional)
    stride = patch_size // 2  # 50% overlap if stride = patch_size // 2

    height, width = work_image.shape

    # Calculate total number of patches for the progress bar
    num_patches_y = ((height - patch_size) // stride) + 1
    num_patches_x = ((width - patch_size) // stride) + 1
    total_patches = num_patches_y * num_patches_x

    data_list = []
    patches = []
    positions = []

    # Collect patches with progress indicator
    for y in tqdm(range(0, height - patch_size + 1, stride), desc="Processing rows"):
        for x in range(0, width - patch_size + 1, stride):
            patch = work_image[y: y + patch_size, x: x + patch_size]
            patch_resized = cv2.resize(
                patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )
            patch_normalized = patch_resized / 255.0  # Normalize

            # Ignore black areas around the sample
            if np.any(patch_normalized == 0):
                continue

            patches.append(patch_normalized)
            positions.append((x, y))

    # Convert patches to a NumPy array and adjust dimensions
    patches_array = np.array(patches)
    if len(patches_array.shape) == 3:
        # Add channel dimension if needed (e.g., for grayscale images)
        patches_array = np.expand_dims(patches_array, axis=-1)

    # Perform predictions in batches with progress indicator
    batch_size = 32  # Adjust based on available memory
    predictions = []
    num_batches = int(np.ceil(len(patches_array) / batch_size))

    for i in tqdm(range(num_batches), desc="Predicting patches"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_patches = patches_array[start_idx:end_idx]
        batch_predictions = model.predict(batch_patches)
        predictions.extend(batch_predictions)

    # Collect results
    for idx, (x, y) in enumerate(positions):
        prediction = predictions[idx]
        row = [x, y, prediction]
        data_list.append(row)

    # Write results to CSV
    write_defect_data(
        filename=image_path, patch_size=patch_size, stride=stride, data_list=data_list
    )


def main():
    defect_recognition_parallel_computing(image_path="predictionDataCSV/20240926_A1-1/20240926_A1-1.bmp", model_name="Model_20241127")
    defect_recognition_parallel_computing(image_path="predictionDataCSV/20240926_A2-1/20240926_A2-1.bmp", model_name="Model_20241127")
    
if __name__ == "__main__":
    main()

