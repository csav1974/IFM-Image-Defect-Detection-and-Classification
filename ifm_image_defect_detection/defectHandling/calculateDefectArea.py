import numpy as np
import cv2
import os
from ifm_image_defect_detection.enumDefectTypes import DefectType
from ifm_image_defect_detection.defectHandling.chippingDefectHandling import calculate_defect_map_chipping
from ifm_image_defect_detection.defectHandling.whiskersDefectHandling import calculate_defect_map_whiskers
from ifm_image_defect_detection.defectHandling.calculate_mean_background import list_to_mean_noise as mean_noise


def calculate_defect_area_fromList(image, data_list, patch_size = 32,):

    save_image = False

    defect_maps = []
    mean_background_value = mean_noise(image, data_list[-1][0][:1000], patch_size) # data_list[-1][0] is a array with all no_defect positions
    data_list.pop(-1)
    for defect_list, defect_type in data_list:
        defect_map, num_non_probe_area = list_to_defect_map(
            image=image, patch_size=patch_size, data_list=defect_list, defect_type=defect_type, background_value=mean_background_value
        )
        if defect_type == DefectType.CHIPPING :
            chipping_area = np.sum(defect_map == 0)
        if defect_type == DefectType.WHISKERS :
            whiskers_area = np.sum(defect_map == 0)
        if defect_type == DefectType.SCRATCHES :
            scratches_area = np.sum(defect_map == 0)
        defect_maps.append((defect_map, defect_type))



    only_defect_maps = [defect_map for defect_map, _ in defect_maps]


    stacked_maps = np.stack(only_defect_maps, axis=0)

    combined_map = np.all(stacked_maps == 1, axis=0).astype(int)

    num_zeros = np.sum(combined_map == 0)
    num_ones = np.sum(combined_map == 1) - num_non_probe_area

    # calculate defect area
    if num_ones == 0:
        ratio = float("inf")
    else:
        ratio = (num_zeros / num_ones) * 100

    print_results(num_zeros, num_ones, ratio)

    if save_image:
        save_image_with_defects(defect_maps, image)


    return whiskers_area, chipping_area, scratches_area, num_zeros, num_ones, ratio

def list_to_defect_map(image, patch_size, data_list, defect_type, background_value):



    num_non_probe_area = np.sum(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) == 0)
    if (defect_type == DefectType.CHIPPING 
        or defect_type == DefectType.SCRATCHES) :
        defect_map = calculate_defect_map_chipping(
            data_list, image, threshold=0.55, patch_size=patch_size
        )       
    if defect_type == DefectType.WHISKERS:
        defect_map = calculate_defect_map_whiskers(
            data_list, image, threshold=0.17, patch_size=patch_size, background_value=background_value
        )

    return defect_map, num_non_probe_area

def save_image_with_defects(maps, image):

    for map in maps:
        mask = map[0] == 0
        if (map[1] == DefectType.CHIPPING):
            image[mask] = (0, 0, 255)
        if (map[1] == DefectType.WHISKERS):
            image[mask] = (255, 0, 0)
        if (map[1] == DefectType.SCRATCHES):
            image[mask] = (0, 255, 255)


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

    name_for_saved_image = "detectedErrorsShown.bmp"
    folder_for_example_Images = "exampleImage"
    final_name = os.path.join(folder_for_example_Images, name_for_saved_image)
    cv2.imwrite(final_name, image)

    print(f"defects save to {final_name}")

def print_results(num_zeros, num_ones, ratio):
    print(f"\nNumer of defect Pixels: {num_zeros}")
    print(f"Number of working Pixels: {num_ones}")
    print(f"Ratio of defect to working: {ratio:.2f}%")


########
# From here on the rest of the code is no longer used.
# Will probably be deleted in the future
#########

# import csv
# from csvHandling.readFromPredictionCSV import read_from_csv
# from defectHandling.whiskersDefectHandling import calculate_unknown_defect_area

# def find_csv_files(directory):
#     csv_files = []

#     # Iterate over all files in the specified directory
#     for file in os.listdir(directory):
#         # Check if the file has a .csv extension
#         if file.endswith(".csv"):
#             # Create the full path of the file
#             full_path = os.path.join(directory, file)
#             # Add the path to the list
#             csv_files.append(full_path)

#     return csv_files

# def calculate_defect_area_fromCSV(image_path, csv_folder):
#     save_image = True
#     save_results = True

#     csv_path_list = find_csv_files(csv_folder)
#     image = cv2.imread(image_path)
#     defect_maps = []
#     for file_path in csv_path_list:
#         defect_type = read_from_csv(file_path)[4]

#         defect_map, num_non_probe_area = csv_to_defect_map(
#             image=image, csv_path=file_path
#         )
#         if defect_type == DefectType.CHIPPING :
#             chipping_area = np.sum(defect_map == 0)
#         if defect_type == DefectType.WHISKERS :
#             whiskers_area = np.sum(defect_map == 0)
#         if defect_type == DefectType.SCRATCHES :
#             scratches_area = np.sum(defect_map == 0)
#         defect_maps.append((defect_map, defect_type))

#     # defect_maps.append(calculate_unknown_defect_area(image)) #test



#     # Extract only the defect maps from the array
#     only_defect_maps = [defect_map for defect_map in defect_maps]
#     stacked_maps = np.stack(only_defect_maps, axis=0)

#     # Use np.all to check if all maps have 1 at each position
#     combined_map = np.all(stacked_maps == 1, axis=0).astype(int)

#     num_zeros = np.sum(combined_map == 0)
#     num_ones = np.sum(combined_map == 1) - num_non_probe_area

#     # calculate defect area
#     if num_ones == 0:
#         ratio = float("inf")
#     else:
#         ratio = (num_zeros / num_ones) * 100

#     print_results(num_zeros, num_ones, ratio)

#     if save_image:
#         save_image_with_defects(defect_maps, image)

#     if save_results:
#         save_results_to_CSV()


#     def save_results_to_CSV():
#         csv_filename = os.path.splitext(os.path.split(image_path)[-1])[0]

#         folder_path = "defectAreaCSV"
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         file_path = os.path.join(folder_path, f"{csv_filename}.csv")

#         with open(file_path, mode="w", newline="") as file:
#             writer = csv.writer(file)
#             # Header
#             writer.writerow(
#                 [
#                     "Image_Name",
#                     "Chipping_Area_Total",
#                     "Chipping_Area_Relativ",
#                     "Whiskers_Area_Total",
#                     "Whiskers_Area_Relativ",
#                     "Scratches_Area_Total",
#                     "Scratches_Area_Relativ",
#                     "Defect_Area_Total",
#                     "Defect_Area_Relativ"
#                 ]
#             )
#             writer.writerow(
#                 [os.path.splitext(os.path.split(image_path)[-1])[0], chipping_area, (chipping_area / num_ones) * 100, whiskers_area, (whiskers_area / num_ones) * 100, scratches_area, (scratches_area / num_ones) * 100, num_zeros, ratio]
#             )


#         print(f"CSV-Datei wurde erfolgreich unter {file_path} gespeichert.")
#     return ratio

# def csv_to_defect_map(image, csv_path):
#     start_x, start_y, image_size, patch_size, defect_type, image_name, coordinates = (
#         read_from_csv(csv_path)
#     )
#     if image_size == 0:
#         image_resized = image
#     else:
#         image_resized = image[
#             start_y : start_y + image_size, start_x : start_x + image_size
#         ]
#     num_non_probe_area = np.sum(cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) == 0)
#     if (defect_type == DefectType.CHIPPING 
#         or defect_type == DefectType.SCRATCHES) :
#         defect_map = calculate_defect_map_chipping(
#             coordinates, image_resized, threshold=0.9
#         )       
#     if defect_type == DefectType.WHISKERS:
#         defect_map = calculate_defect_map_whiskers(
#             coordinates, image_resized, threshold=0.1
#         )

#     return defect_map, num_non_probe_area