import numpy as np
import cv2
import os
from readFromCSV import read_from_csv
from enumDefectTypes import DefectType
from defectHandling.chippingDefectHandling import calculate_defect_map_chipping
from defectHandling.whiskersDefectHandling import calculate_defect_map_whiskers
from defectHandling.whiskersDefectHandling import calculate_unknown_defect_area


def calculate_defect_area(image_path, csv_folder):
    safe_image = True

    csv_path_list = find_csv_files(csv_folder)
    image = cv2.imread(image_path)
    defect_maps = []
    for file_path in csv_path_list:
        defect_map, num_non_probe_area = csv_to_defect_map(
            image=image, csv_path=file_path
        )
        defect_maps.append(defect_map)

    # defect_maps.append(calculate_unknown_defect_area(image)) #test

    stacked_maps = np.stack(defect_maps, axis=0)

    # Use np.all to check if all maps have 1 at each position
    combined_map = np.all(stacked_maps == 1, axis=0).astype(int)

    num_zeros = np.sum(combined_map == 0)
    num_ones = np.sum(combined_map == 1) - num_non_probe_area

    # calculate defect area
    if num_ones == 0:
        ratio = float("inf")
    else:
        ratio = (num_zeros / num_ones) * 100

    print_results(num_zeros, num_ones, ratio)

    return ratio


def safe_image_with_defects(combined_map, image):
    mask = combined_map == 0
    mask = np.transpose(mask)
    image[mask] = 0

    # # scales Picture for output
    # width_resized = 4000
    # height_resized = int(
    #     (width_resized / image.shape[1]) * image.shape[0]
    # )  # scaling height to width
    # resized_image = cv2.resize(image, (width_resized, height_resized))

    # cv2.imshow("Detected Defects", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # #####

    name_for_safed_image = "detectedErrorsShown.bmp"
    folder_for_example_Images = "exampleImage"
    final_name = os.path.join(folder_for_example_Images, name_for_safed_image)
    cv2.imwrite(final_name, image)


def csv_to_defect_map(image, csv_path):
    start_x, start_y, image_size, patch_size, defect_type, image_name, coordinates = (
        read_from_csv(csv_path)
    )
    if image_size == 0:
        image_resized = image
    else:
        image_resized = image[
            start_y : start_y + image_size, start_x : start_x + image_size
        ]
    num_non_probe_area = np.sum(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) == 0)
    if defect_type == DefectType.CHIPPING:
        defect_map = calculate_defect_map_chipping(
            coordinates, image_resized, threshold=0.9
        )
    if defect_type == DefectType.WHISKERS:
        defect_map = calculate_defect_map_whiskers(
            coordinates, image_resized, threshold=0.1
        )

    return defect_map, num_non_probe_area


def find_csv_files(directory):
    csv_files = []

    # Iterate over all files in the specified directory
    for file in os.listdir(directory):
        # Check if the file has a .csv extension
        if file.endswith(".csv"):
            # Create the full path of the file
            full_path = os.path.join(directory, file)
            # Add the path to the list
            csv_files.append(full_path)

    return csv_files


def print_results(num_zeros, num_ones, ratio):
    print(f"\nNumer of defect Pixels: {num_zeros}")
    print(f"Number of working Pixels: {num_ones}")
    print(f"Ratio of defect to working: {ratio:.2f}%")


calculate_defect_area("sampleOnlyBMP/20240610_A6-2m_10x$3D.bmp", "testing")
