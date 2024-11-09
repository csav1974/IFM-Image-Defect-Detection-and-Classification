from enumDefectTypes import DefectType
from shapely.geometry import Polygon

def calculate_defect_count(polygon_data, defect_type : DefectType) :
    # checking if polygon_data is of the correct type
    if not isinstance(polygon_data, list):
        raise ValueError("merged_polygons must be a list")

    for item in polygon_data:
        if not isinstance(item, dict):
            raise ValueError("Each item in merged_polygons must be a dictionary")

        if not isinstance(item.get('polygon'), Polygon):
            raise ValueError("Each 'polygon' in merged_polygons must be a shapely Polygon")

        if not isinstance(item.get('color'), tuple):
            raise ValueError("Each 'color' in merged_polygons must be a tuple")

        if 'defect_type' not in item:
            raise ValueError("Each item in merged_polygons must contain 'defect_type'")
            
    poly_of_given_defect = [poly for poly in polygon_data if poly['defect_type'] == defect_type]
    if defect_type == DefectType.CHIPPING:
        return len(poly_of_given_defect)

    polygon_count = len(poly_of_given_defect)

    # If no polygons, return 0
    if polygon_count == 0:
        return 0

    # Step 1: Calculate the area of all polygons
    polygon_areas = []
    for item in poly_of_given_defect:
        polygon = item['polygon']
        polygon_areas.append(polygon.area)

    # Step 2: Sort the areas in ascending order
    polygon_areas.sort()

    # Step 3: Calculate how many polygons represent the smallest 25%
    smallest_25_percent_count = max(1, int(polygon_count * 0.25))  # At least 1 polygon
    
    # Step 4: Select the smallest 10% of polygons
    smallest_polygons = polygon_areas[:smallest_25_percent_count]

    # Step 5: Sum the area of the smallest 10% of polygons
    total_area_of_smallest = sum(smallest_polygons)

    # Step 6: Calculate the average area of the smallest polygons
    average_area_of_smallest = total_area_of_smallest / smallest_25_percent_count


    # Perform integer division of each polygon's area by the average area
    division_sum = 0
    for item in poly_of_given_defect:
        polygon = item['polygon']
        polygon_area = polygon.area
        division_result = polygon_area // average_area_of_smallest  # Integer division
        if division_result == 0:
            division_result = 1
        division_sum += division_result

   
    return int(division_sum)