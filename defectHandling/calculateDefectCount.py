from enumDefectTypes import DefectType
from shapely.geometry import Polygon
from shapely import vectorized
import numpy as np


def calculate_defect_count_old(polygon_data, defect_type : DefectType) :
    """
    Calculates the defect count for a given defect type in a list of polygons.

    This function analyzes a list of polygon data, verifies its structure, 
    and calculates the number of defects of the specified type. 
    For general defect types, the calculation involves the smallest 25% 
    of polygons based on their area.

    Parameters:
    ----------
    polygon_data : list
        A list of dictionaries, each containing:
        - 'polygon': A `shapely.geometry.Polygon` object.
        - 'color': A tuple representing the polygon's color.
        - 'defect_type': The type of defect (DefectType).

    defect_type : DefectType
        The type of defect to count (e.g., CHIPPING).

    Returns:
    -------
    int
        The calculated defect count:
        - If the defect type is `CHIPPING`, returns the number of matching polygons.
        - Otherwise, calculates based on the area of the smallest polygons.

    Raises:
    ------
    ValueError:
        If `polygon_data` is improperly structured or contains invalid entries.
    """

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
    
    # Step 4: Select the smallest 25% of polygons
    smallest_polygons = polygon_areas[:smallest_25_percent_count]

    # Step 5: Sum the area of the smallest 25% of polygons
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


def calculate_defect_count_new(polygon_data, defect_type: DefectType, defect_map):
    """
    Calculates the defect count for a given defect type based on the number of black pixels (0,0,0)
    within the polygons defined in defect_map.

    For each polygon of the specified defect type, the function counts the number of black pixels
    within that polygon. Then, it selects the 25% of polygons with the smallest area and computes
    the average black pixel count among them. Finally, for each polygon, the function performs an integer
    division of its black pixel count by this average (with a minimum value of 1) and sums the results.

    Parameters:
    ----------
    polygon_data : list
        A list of dictionaries, each containing:
            - 'polygon': A shapely.geometry.Polygon object.
            - 'color': A tuple representing the polygon's color.
            - 'defect_type': The defect type (DefectType).
    defect_type : DefectType
        The defect type to consider.
    defect_map : numpy.ndarray
        The image where the polygons are defined. Black pixels have the value (0,0,0).

    Returns:
    -------
    int
        The calculated defect count.
    """
    # Validate input data
    if not isinstance(polygon_data, list):
        raise ValueError("polygon_data must be a list")
    for item in polygon_data:
        if not isinstance(item, dict):
            raise ValueError("Each item in polygon_data must be a dictionary")
        if not isinstance(item.get('polygon'), Polygon):
            raise ValueError("Each 'polygon' in polygon_data must be a shapely Polygon")
        if not isinstance(item.get('color'), tuple):
            raise ValueError("Each 'color' in polygon_data must be a tuple")
        if 'defect_type' not in item:
            raise ValueError("Each item in polygon_data must contain 'defect_type'")
    
    # Filter polygons with the specified defect type
    poly_of_given_defect = [item for item in polygon_data if item['defect_type'] == defect_type]
    polygon_count = len(poly_of_given_defect)
    if polygon_count == 0:
        return 0
    if defect_type == DefectType.CHIPPING:
        return polygon_count

    def count_black_pixels_in_polygon(polygon, defect_map):
        """
        Counts the number of black pixels (0,0,0) within a given polygon.
        Uses vectorized operations over the polygon's bounding box for efficiency.
        """
        # Get the bounding box of the polygon and convert to integer indices
        min_x, min_y, max_x, max_y = polygon.bounds
        min_x = int(np.floor(min_x))
        min_y = int(np.floor(min_y))
        max_x = int(np.ceil(max_x))
        max_y = int(np.ceil(max_y))
        
        # Create a grid of x and y coordinates within the bounding box
        xs, ys = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
        
        # Use vectorized function to check which points are inside the polygon
        # This returns a boolean array with True for points inside the polygon
        
        inside_mask = vectorized.contains(polygon, xs, ys)
        
        # Extract the region of interest from the defect_map
        region = defect_map[min_y:max_y, min_x:max_x]

        # Create a boolean mask for black pixels (0,0,0) within the region
        black_mask = (region == 0)
        
        # Combine the masks to count only black pixels inside the polygon
        count = np.sum(inside_mask & black_mask)
        
        
        return int(count)

    # Create a list of tuples (polygon_area, black_pixel_count) for each polygon of the given defect type
    polygon_metrics = []
    for item in poly_of_given_defect:
        polygon = item['polygon']
        area = polygon.area
        black_count = count_black_pixels_in_polygon(polygon, defect_map)
        if black_count > area * 0.01:
            polygon_metrics.append((area, black_count))


    # Sort the list by polygon area in ascending order
    polygon_metrics.sort(key=lambda x: x[0])

    # Determine the number of polygons representing the smallest 25% by area
    smallest_25_percent_count = max(1, int(polygon_count * 0.25))
    smallest_polygons = polygon_metrics[:smallest_25_percent_count]


    # Calculate the average black pixel count of the area-wise smallest polygons
    total_black_smallest = sum(black_count for _, black_count in smallest_polygons)
    average_smallest = total_black_smallest / smallest_25_percent_count




    # Compute the defect count by performing an integer division of each polygon's black pixel count by the average,
    # ensuring a minimum result of 1, and then summing the results.
    division_sum = 0
    for _, black_count in polygon_metrics:
        division_result = int(black_count // average_smallest)
        if division_result == 0:
            division_result = 1
        division_sum += division_result

    return int(division_sum)

