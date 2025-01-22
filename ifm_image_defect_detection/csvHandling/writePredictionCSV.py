import os
import csv
from ifm_image_defect_detection.enumDefectTypes import DefectType

def write_defect_data(filename, 
                      patch_size, 
                      stride, 
                      data_list, 
                      defect_type : DefectType = None,
                      ):
    # image_name = os.path.splitext(os.path.split(filename)[-1])[0]
    # folder_name = os.path.join("predictionDataCSV", image_name)
    path = os.path.splitext(filename)[0]
    if defect_type :
        file_path = f"{path}_{defect_type.value}.csv"
    else :
        file_path = f"{path}_prediction.csv"

    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # write Metadata
        writer.writerow(["Image Name", "Patch Size", "Stride", "Defect Type"])
        if defect_type:
            writer.writerow([filename, patch_size, stride, defect_type.value])
        else:
            writer.writerow([filename, patch_size, stride, "None"])
        # write Header
        writer.writerow(
            ["x", "y", "Whiskers", "Chipping", "Scratch", "No Error"]
        )

        for row in data_list:
            x, y, prediction = row
            # round data an write to csv
            formatted_prediction = [f"{p:.5f}" for p in prediction]
            writer.writerow([x, y] + formatted_prediction)
    print(f"CSV-File saved as {file_path}")