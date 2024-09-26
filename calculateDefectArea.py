import numpy as np
import cv2
import os
import csv
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
        defect_type = read_from_csv(file_path)[4]

        defect_map, num_non_probe_area = csv_to_defect_map(
            image=image, csv_path=file_path
        )
        if defect_type == DefectType.CHIPPING :
            chipping_area = np.sum(defect_map == 0)
        if defect_type == DefectType.WHISKERS :
            whiskers_area = np.sum(defect_map == 0)
        if defect_type == DefectType.SCRATCHES :
            scratches_area = np.sum(defect_map == 0)
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

    if safe_image:
        safe_image_with_defects(combined_map, image)


    def save_results_to_CSV():
        csv_filename = os.path.splitext(os.path.split(image_path)[-1])[0]

        folder_path = "defectAreaCSV"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, f"{csv_filename}.csv")

        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Header
            writer.writerow(
                [
                    "Image_Name",
                    "Chipping_Area_Total",
                    "Chipping_Area_Relativ",
                    "Whiskers_Area_Total",
                    "Whiskers_Area_Relativ",
                    "Scratches_Area_Total",
                    "Scratches_Area_Relativ",
                    "Defect_Area_Total",
                    "Defect_Area_Relativ"
                ]
            )
            writer.writerow(
                [os.path.splitext(os.path.split(image_path)[-1])[0], chipping_area, (chipping_area / num_ones) * 100, whiskers_area, (whiskers_area / num_ones) * 100, scratches_area, (scratches_area / num_ones) * 100, num_zeros, ratio]
            )


        print(f"CSV-Datei wurde erfolgreich unter {file_path} gespeichert.")
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
    if (defect_type == DefectType.CHIPPING 
        or defect_type == DefectType.SCRATCHES) :
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



calculate_defect_area("sampleOnlyBMP/20240424_A2-2m$3D_10x.bmp", "testing/20240424_A2-2m$3D_10x")
