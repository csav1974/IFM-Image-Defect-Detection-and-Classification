import numpy as np
import cv2

def create_ones_array_from_image(image, chipping_position_list, threshold):

    # convert image to np-array
    image_np = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    # create np-array like image for storing defect location
    ones_array = np.ones_like(image_np)
    
    ###testing
    meanvalue_array = []
    ####
    for (x, y, patch_size) in chipping_position_list:
        defect_area = image[y : y + patch_size, x : x + patch_size]
        ###
        meanvalue = np.array(defect_area).mean()
        meanvalue_array.append(meanvalue)
        ###
        print(f"maximum Brightness chipping defect: {meanvalue}")
        for i in range(defect_area.shape[0]):  # iterating over hight
            for j in range(defect_area.shape[1]):  # iterating over width
                # calculate Brightness of each Pixel
                pixel_value = defect_area[i, j]
                brightness = np.mean(pixel_value)  # mean over RGB
                
                if brightness > threshold:
                    ones_array[i, j] = 0
    ####
    absolutmin = min(meanvalue_array)
    print(f"absulutmin of mean: {absolutmin}")
    ####
    return ones_array