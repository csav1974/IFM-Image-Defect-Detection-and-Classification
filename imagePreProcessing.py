import cv2 as cv
import numpy as np
import os




# Transform an IFM Image to a blurred graylevel Image for further processing.
def TransformToBlurredGray(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ksize = (5, 5)  # Kernelgröße
    sigmaX = 1.0    # Standardabweichung in X-Richtung
    blurred_image = cv.GaussianBlur(gray, ksize, sigmaX)
    return blurred_image

# use HoughCircles to find only the probe itself.
# returns a list of circles.
def findCircles(minimalDiameter, maximalDiameter, image):
    circles = cv.HoughCircles(
    image, 
    cv.HOUGH_GRADIENT, dp=3, minDist=3000,
    param1=150, param2=5, minRadius= minimalDiameter // 2, maxRadius= maximalDiameter // 2
    )
    return circles


# takes the image and the blurred Version and matches it to the original image to create a image thats only the Probe with anything else black
def bmpToProbeOnly(filename):
    image=cv.imread(filename)
    blurredImage = TransformToBlurredGray(image)
    circles = findCircles(3500,5000, blurredImage)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        center = (circles[0, 0][0], circles[0, 0][1])
        radius = circles[0, 0][2]

        # cv.circle(image, center, radius + 5, (0, 255, 0), 4)  # Umrandung des Kreises

        # cuts out the square of the Probe
        probeOnlyImage = image[center[1]-(radius):center[1]+(radius), center[0]-(radius):center[0]+(radius)]

        # makes anything but the Probe black
        height, width = probeOnlyImage.shape[:2]
        center = (width // 2, height // 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.circle(mask, center, radius, (255), thickness=-1)
        masked_image = cv.bitwise_and(probeOnlyImage, probeOnlyImage, mask=mask)
        b, g, r = cv.split(masked_image)
        alpha_channel = mask  
        return cv.merge((b, g, r, alpha_channel))


def saveBMPtoFolder(image, input_folder):
    """
    Saves a BMP image to a specified folder with a unique filename.
    If 'probeOnly_0.bmp' already exists, it appends a number to the filename.

    Args:
        image: The image to save.

    Returns:
        str: The full path to the saved file.
    """
    folder = 'sampleOnlyBMP'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # create Filename based on the folder it comes from
    base_filename = os.path.basename(input_folder)
    extension = '.bmp'
    filename = os.path.join(folder, f'{base_filename}{extension}')

    print(f"saved to {filename}")
    # Save the image
    cv.imwrite(filename, image)
    
    return filename
    
def finishedProcessing(filename, folderpath):
    progressed_filename = saveBMPtoFolder(image= bmpToProbeOnly(filename), input_folder= folderpath)
    return progressed_filename


# # Bild auf gewünschte Größe skalieren
# width = 1000  # gewünschte Breite
# height = int((width / image.shape[1]) * image.shape[0])  # Höhe entsprechend skalieren
# resized_image = cv.resize(image, (width, height))

# cv.imshow('Detected Defects', resized_image)
# cv.waitKey(0)
# cv.destroyAllWindows()
