import cv2 as cv
import numpy as np
import os
import imagePreProcessing 

# Load picture and transform to blurred gray
folder = 'sampleOnlyBMP'
filename = os.path.join(folder, f'probeOnly.bmp')
image = cv.imread(filename)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ksize = (5, 5)  # Kernelgröße
sigmaX = 1.0    # Standardabweichung in X-Richtung
blurred_image = cv.GaussianBlur(gray, ksize, sigmaX)





# use HoughCircles to find possible defects
def findCircles(minimalDiameter, maximalDiameter):
    circles = cv.HoughCircles(
    blurred_image, 
    cv.HOUGH_GRADIENT, dp=1, minDist=20,
    param1=50, param2=27, minRadius= int(minimalDiameter / 2), maxRadius= int(maximalDiameter / 2)
    )
    return circles


def merge_and_remove_duplicates(circles1, circles2):
    # check if circles were found
    if circles1 is not None:
        circles1 = np.uint16(np.around(circles1))
    if circles2 is not None:
        circles2 = np.uint16(np.around(circles2))

    # merge arrays
    if circles1 is not None and circles2 is not None:
        combined_circles = np.concatenate((circles1, circles2), axis=1)
    elif circles1 is not None:
        combined_circles = circles1
    elif circles2 is not None:
        combined_circles = circles2
    else:
        return None  # if both are empty

    # removing duplicates
    if combined_circles.size == 0:
        return None

    # check if center and radius is the same. If yes, remove one
    unique_circles = np.unique(combined_circles, axis=1, return_index=True)[1]
    unique_circles = combined_circles[:, unique_circles]

    return unique_circles

# create an Array of possible defects with different parameters to cover more possibilitys

def parameterRange(minimalDiameter, maximalDiameter):
    startDiameter = 19
    uniqueCircles = findCircles(minimalDiameter, startDiameter)
    for currentMaxDiameter in range(startDiameter + 1, maximalDiameter):
        currentCircles = findCircles(int(currentMaxDiameter *0.1), currentMaxDiameter)
        uniqueCircles = merge_and_remove_duplicates(uniqueCircles, currentCircles)
   
    return uniqueCircles



# saves the found defects 
def saveCircleROIsToBMP(circle_rois, folder='detectedErrors'):
    # creates a folder if it doesn´t already exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for idx, roi in enumerate(circle_rois):
        filename = os.path.join(folder, f'circle_roi_{idx + 1}.bmp')
        cv.imwrite(filename, roi)
    print(f"Alle ROIs wurden im Ordner '{folder}' als .BMP-Dateien gespeichert.")

# draws circles for visual representation and uses saveCircleROIsToBMP to save the defects for later use
def checkAndDrawCircles(workImage, circles, minimalDiameter, maximalDiameter, maximalMeanIntensity, imageCopy):
    circle_rois = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        errorCount = 0
        for i in circles[0, :]:
            center = (i[0], i[1])  # Mittelpunkt des Kreises
            radius = i[2]  # Radius des Kreises
            
            # Schneide den Bereich um den Kreis aus
            circle_roi = workImage[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
            circle_roi_save = imageCopy[center[1]-(radius+3):center[1]+(radius+3), center[0]-(radius+3):center[0]+(radius+3)]
            # Berechne den Mittelwert und die Standardabweichung innerhalb des Kreises
            mean_intensity = np.mean(circle_roi)
            std_intensity = np.std(circle_roi)

            # Prüfe, ob der Mittelwert der Intensität innerhalb des Kreises niedrig ist (dunkel) und der Durchmesser im gewünschten Bereich liegt
            if mean_intensity < maximalMeanIntensity and minimalDiameter <= 2 * radius <= maximalDiameter:
                errorCount += 1
                cv.circle(image, center, radius + 5, (0, 255, 0), 4)  # Umrandung des Kreises
                circle_rois.append(circle_roi_save)  # ROI hinzufügen

    saveCircleROIsToBMP(circle_rois)  # Speichern der ROIs
    return errorCount

# print on console to check the parameters and how they perform. 
# only use in combinaton with visual feedback. More found defects doesn´t automatically mean better performance.
def errorCountChecker(absolutMinimalDiameter, absolutMaximalDiameter, absolutMaximalMeanIntensity):

    print("Number of Errors Detected with \n\t minimal Diameter = ", absolutMinimalDiameter, "\n\t maximal Diameter = ", absolutMaximalDiameter, "\n\t maximal Mean Intensity = ", absolutMaximalMeanIntensity, "\n\t Errors =", numberOfErrorsDetected)




# defining parameters
absolutMinimalDiameter = 6
absolutMaximalDiameter = 60
absolutMaximalMeanIntensity = 400


circles = parameterRange(absolutMinimalDiameter, absolutMaximalDiameter)
numberOfErrorsDetected = checkAndDrawCircles(workImage= blurred_image, circles= circles,minimalDiameter= absolutMinimalDiameter,maximalDiameter= absolutMaximalDiameter,maximalMeanIntensity= absolutMaximalMeanIntensity, imageCopy= image)
errorCountChecker(absolutMinimalDiameter, absolutMaximalDiameter, absolutMaximalMeanIntensity)



# scales Picture for output
width = 800  # gewünschte Breite
height = int((width / image.shape[1]) * image.shape[0])  # Höhe entsprechend skalieren
resized_image = cv.resize(image, (width, height))

# shows picture
cv.imshow('Detected Defects', resized_image)
cv.waitKey(0)
cv.destroyAllWindows()
