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
        next(reader)  # Skip first Header
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
        next(reader)  # Skip second Header
        for row in reader:
            data_list.append(parse_csv_row(row))

    return (
        image_name,
        patch_size,
        stride,
        defect_type,  
        data_list,
    )