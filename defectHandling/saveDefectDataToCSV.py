import os
import csv



def save_results_to_CSV(folder_path, whiskers_area, chipping_area, scratches_area, num_zeros, num_ones, ratio, whisker_count, chipping_count):
    sample_name = os.path.split(folder_path)[-1]


    file_path = os.path.join(folder_path, f"{sample_name}_defectData.csv")

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Header
        writer.writerow(
            [
                "Image_Name",
                "Whiskers_Area_Total",
                "Whiskers_Count",
                "Whiskers_Area_Relativ",
                "Chipping_Area_Total",
                "Chipping_Count",
                "Chipping_Area_Relativ",
                "Scratches_Area_Total",
                "Scratches_Area_Relativ",
                "Defect_Area_Total",
                "Defect_Area_Relativ"
            ]
        )
        writer.writerow(
            [sample_name, whiskers_area, whisker_count, (whiskers_area / num_ones) * 100, chipping_area, chipping_count, (chipping_area / num_ones) * 100, scratches_area, (scratches_area / num_ones) * 100, num_zeros, ratio]
        )


    print(f"CSV-Datei wurde unter {file_path} gespeichert.")
