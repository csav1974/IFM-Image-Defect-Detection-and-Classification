import numpy as np

"""
    the code here is just placeholder.
    the actual detection which part of the Whiskers defect images is really a defect will later be added.
"""


def create_ones_array_from_image(image, whiskers_position_list, patch_size, threshold):

    # convert image to np-array
    image_np = np.array(image)
    
    # create np-array like image for storing defect location
    ones_array = np.ones_like(image_np[...,0])
    
    for (x, y) in whiskers_position_list:
        defect_area = image[y : y + patch_size, x : x + patch_size]
        for i in range(defect_area.shape[0]):  # iterating over hight
            for j in range(defect_area.shape[1]):  # iterating over width
                # calculate Brightness of each Pixel
                pixel_value = defect_area[i, j]
                brightness = np.mean(pixel_value)  # mean over RGB
                
                if brightness > threshold:
                    ones_array[i, j] = 0


    return ones_array