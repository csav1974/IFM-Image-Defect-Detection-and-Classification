import os
import csv

def save_results_to_CSV(
    folder_path,
    whiskers_area,
    chipping_area,
    scratches_area,
    num_zeros,
    num_ones,
    ratio,
    whisker_count,
    chipping_count,
    whiskers_positions = [],
    chipping_positions = []  
):
    sample_name = os.path.split(folder_path)[-1]

    file_path = os.path.join(folder_path, f"{sample_name}_defectData.csv")

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(["Image_Name", sample_name])
        writer.writerow(["Whiskers_Area_mm²", whiskers_area])
        writer.writerow(["Whiskers_Count", whisker_count])
        writer.writerow(["Whiskers_Area_Relativ", (whiskers_area / num_ones) * 100])
        writer.writerow(["Chipping_Area_mm²", chipping_area])
        writer.writerow(["Chipping_Count", chipping_count])
        writer.writerow(["Chipping_Area_Relativ", (chipping_area / num_ones) * 100])
        writer.writerow(["Scratches_Area_mm²", scratches_area])
        writer.writerow(["Scratches_Area_Relativ", (scratches_area / num_ones) * 100])
        writer.writerow(["Defect_Area_mm²", num_zeros])
        writer.writerow(["Defect_Area_Relativ", ratio])

        writer.writerow(["Whiskers_Position", str(whiskers_positions)])
        writer.writerow(["Chipping_Positions", str(chipping_positions)])
        
    print(f"CSV-Datei wurde unter {file_path} gespeichert.")
