import os
import csv
from enumDefectTypes import DefectType


def read_from_csv(csv_path):
    coordinates = []

    with open(csv_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(
            reader
        )  # skip first header line (start_x,start_y,image_size,patch_size,image_name)
        first_row = next(reader)
        start_x, start_y, image_size, patch_size, defect_type, filename = first_row
        start_x, start_y, image_size, patch_size = (
            int(start_x),
            int(start_y),
            int(image_size),
            int(patch_size),
        )
        if defect_type == DefectType.CHIPPING.value:
            defect_type = DefectType.CHIPPING
        if defect_type == DefectType.WHISKERS.value:
            defect_type = DefectType.WHISKERS
        if defect_type == DefectType.NO_ERROR.value:
            defect_type = DefectType.NO_ERROR
        image_name = os.path.splitext(os.path.split(filename)[-1])[0]
        next(reader)  # to skip values
        next(reader)  # to skip second header (x,y)
        for row in reader:
            coordinates.append(
                (int(row[0]), int(row[1]), int(row[2]))
            )  # reads x, y, patchsize

    return (
        start_x,
        start_y,
        image_size,
        patch_size,
        defect_type,
        image_name,
        coordinates,
    )
