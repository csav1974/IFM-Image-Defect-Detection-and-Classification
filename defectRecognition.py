import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from enumDefectTypes import DefectType

######
# Old Version, not used anymore
######



# shows all areas that were marked as defects as squares on RGB Probe-Image
def visual_representation(image, defect_positions_chipping, defect_positions_whiskers, defect_positions_scratches):
    # Print positions to debug
    # print(f"Chipping Defects: {defect_positions_chipping}")
    # print(f"Whiskers Defects: {defect_positions_whiskers}")

    # Get image dimensions
    height, width = image.shape[:2]

    # Draw rectangles for whiskers defects
    for x, y, patch_size in defect_positions_whiskers:
        if 0 <= x < width and 0 <= y < height:
            cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), (0, 0, 0), 2)
        else:
            print(f"Skipping out-of-bounds defect at: {x}, {y}")

    # Draw rectangles for chipping defects
    for x, y, patch_size in defect_positions_chipping:
        if 0 <= x < width and 0 <= y < height:
            cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), (0, 0, 255), 2)
        else:
            print(f"Skipping out-of-bounds defect at: {x}, {y}")

    # Draw rectangles for scratches defects
    for x, y, patch_size in defect_positions_scratches:
        if 0 <= x < width and 0 <= y < height:
            cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), (0, 255, 0), 2)
        else:
            print(f"Skipping out-of-bounds defect at: {x}, {y}")

    # Scale Picture for output
    width_resized = 600
    height_resized = int((width_resized / image.shape[1]) * image.shape[0])  # scaling height to width
    resized_image = cv2.resize(image, (width_resized, height_resized))

    # Show image with rectangles
    cv2.imshow("Detected Defects", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# safes found defects and non-defect areas to CSV File
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


# Load Model. Assuming model is already trained
model_name = "fullModel_v2"
path_to_model = os.path.join("kerasModels", model_name)
model = tf.keras.models.load_model(f"{path_to_model}.keras")
IMG_SIZE = 32  # scales the patch size down (or up) to 32*32 Pixel

# Load the microscope image
filename = "sampleOnlyBMP/20240424_A2-2m$3D_10x.bmp"
image = cv2.imread(filename)
work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Parameters for running the Programm

show_image = True

safe_image = True

safe_coordinates = True

# set the confident threshold of model prediction
chipping_detection_threshold = 0.60
whiskers_detection_threshold = 0.80
scratches_detection_threshold = 0.60


# Define patch size
patch_size = 120

# Define stride (optional)
stride = patch_size // 2  # 50% overlap if stride = patch_size // 2

start_x = 0
start_y = 0
square_size = 0

# this part is used if only a part of the image should be examed for faster runtime.
# Comment out if you want to exam the whole picture
############

# height, width = work_image.shape
# # Define the size of the square
# square_size = 6000

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

# List to store defect positions
defect_positions = []
defect_positions_chipping = []
defect_positions_whiskers = []
defect_positions_scratches = []
non_defect_position = []
chipping_count = 0
whiskers_count = 0
scratches_count = 0


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

        if (
            prediction[0][0] > whiskers_detection_threshold
            or prediction[0][1] > chipping_detection_threshold
            or prediction[0][2] > scratches_detection_threshold
        ):  # Assuming threshold for defect = 0.3
            defect_positions.append((x, y, patch_size))
            if prediction[0][0] > whiskers_detection_threshold:
                whiskers_count += 1
                defect_positions_whiskers.append((x, y, patch_size))
            if prediction[0][1] > chipping_detection_threshold:
                chipping_count += 1
                defect_positions_chipping.append((x, y, patch_size))
            if prediction[0][2] > scratches_detection_threshold:
                scratches_count += 1
                defect_positions_scratches.append((x, y, patch_size))
        if prediction[0][2] > 0.999:
            non_defect_position.append((x, y, patch_size))
        current_patch_number += 1
        print(f"Patch: {current_patch_number} / {number_of_patches}")
        print(prediction)
        # print(f"Preddiction= {prediction}")
        # print(f"Chipping prediction: {prediction[0]} \nWhiskers prediction: {prediction[1]}")



# Display the result
print(f"\nNumber of Defects found: {len(defect_positions)}")
print(f"Chipping Defects: {chipping_count}\n" f"Scratching Defects: {scratches_count}\n" f"Whiskers Defects: {whiskers_count}")

image_to_show = image
if show_image:
    visual_representation(
        image_to_show,
        defect_positions_chipping=defect_positions_chipping,
        defect_positions_whiskers=defect_positions_whiskers,
        defect_positions_scratches=defect_positions_scratches,
    )

if safe_coordinates:
    common_params = {
        "start_x": start_x,
        "start_y": start_y,
        "square_size": square_size,
        "patch_size": patch_size,
        "filename": filename,
    }
    safe_coordinates_to_CSV(
        defect_positions_chipping,
        defect_type=DefectType.CHIPPING,
        **common_params
    )
    safe_coordinates_to_CSV(
        defect_positions_whiskers,
        defect_type=DefectType.WHISKERS,
        **common_params
    )
    safe_coordinates_to_CSV(
        defect_positions_scratches,
        defect_type=DefectType.SCRATCHES,
        **common_params
    )
    safe_coordinates_to_CSV(
        non_defect_position,
        defect_type=DefectType.NO_ERROR,
        **common_params
    )
if safe_image:
    name_for_safed_image = os.path.split(filename)[-1]
    folder_for_example_Images = "exampleImage"
    final_name = os.path.join(folder_for_example_Images, name_for_safed_image)
    cv2.imwrite(final_name, image_to_show)
