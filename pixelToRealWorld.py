#####
# This code will be changed to automatically take the "pixel_per_square_mm" and "pixel_per_mm" values from the meta data
#####

def pixel_to_square_mm(pixel_count : int, pixel_per_square_mm : float = 316068.84):
    return pixel_count / pixel_per_square_mm

def pixel_to_mm(pixel_count : int, pixel_per_mm : float = 562.2):
    return pixel_count / pixel_per_mm