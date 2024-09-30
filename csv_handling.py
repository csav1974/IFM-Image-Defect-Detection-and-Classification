import os
import csv
from enumDefectTypes import DefectType


def read_from_csv(csv_path):
    data_list = []

    def parse_csv_row(row):
        x = int(row[0])
        y = int(row[1])
        prediction_list = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        return [x, y, prediction_list]

    with open(csv_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Überspringe die erste Kopfzeile
        first_row = next(reader)
        image_name, patch_size, stride, defect_type = first_row
        patch_size, stride = int(patch_size), int(stride)
        if defect_type == DefectType.CHIPPING.value:
            defect_type = DefectType.CHIPPING
        if defect_type == DefectType.WHISKERS.value:
            defect_type = DefectType.WHISKERS
        if defect_type == DefectType.SCRATCHES.value:
            defect_type = DefectType.SCRATCHES
        if defect_type == DefectType.NO_ERROR.value:
            defect_type = DefectType.NO_ERROR
        if defect_type == "None":
            defect_type = None
        next(reader)  # Überspringe die zweite Kopfzeile
        for row in reader:
            data_list.append(parse_csv_row(row))

    return (
        image_name,
        patch_size,
        stride,
        defect_type,  
        data_list,
    )

def write_defect_data(filename, 
                      patch_size, 
                      stride, 
                      data_list, 
                      defect_type : DefectType = None,
                      ):
    image_name = os.path.splitext(os.path.split(filename)[-1])[0]
    folder_name = os.path.join("predictionDataCSV", image_name)
    if defect_type :
        file_path = os.path.join(folder_name, f"{image_name}_{defect_type.value}.csv")
    else :
        file_path = os.path.join(folder_name, f"{image_name}.csv")
    # Erstellen des Ordners, falls er nicht existiert
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Schreiben der Koordinaten in die CSV-Datei
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Schreiben der Metadaten
        writer.writerow(["Image Name", "Patch Size", "Stride", "Defect Type"])
        writer.writerow([filename, patch_size, stride, defect_type.value])

        # Kopfzeile für die Koordinaten und Vorhersagen
        writer.writerow(
            ["x", "y", "Whiskers", "Chipping", "Scratch", "No Error"]
        )

        # Daten in die Datei schreiben
        for row in data_list:
            x, y, prediction = row
            # Runde jede Vorhersage auf 5 Nachkommastellen und schreibe sie in die CSV
            formatted_prediction = [f"{p:.5f}" for p in prediction]
            writer.writerow([x, y] + formatted_prediction)
    print(f"CSV-File saved as {file_path}")