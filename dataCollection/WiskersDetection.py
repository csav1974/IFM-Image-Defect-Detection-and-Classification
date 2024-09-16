import cv2 as cv
import numpy as np
import os
from dataCollection.imagePreProcessing import finishedProcessing as imageProcessing
import dataCollection.filemanagement as filemanagement


# Load picture and transform to blurred gray
def loadPicture(filenameAndPath):
    image = cv.imread(filenameAndPath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ksize = (5, 5)  # Kernelgröße
    sigmaX = 1.0  # Standardabweichung in X-Richtung
    blurred_image = cv.GaussianBlur(gray, ksize, sigmaX)

    return image, blurred_image


# use HoughCircles to find possible defects
def findCircles(minimalDiameter, maximalDiameter, blurred_image):
    circles = cv.HoughCircles(
        blurred_image,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=27,
        minRadius=int(minimalDiameter / 2),
        maxRadius=int(maximalDiameter / 2),
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


def parameterRange(minimalDiameter, maximalDiameter, blurred_image):
    startDiameter = 19
    uniqueCircles = findCircles(minimalDiameter, startDiameter, blurred_image)
    for currentMaxDiameter in range(startDiameter + 1, maximalDiameter):
        currentCircles = findCircles(
            int(currentMaxDiameter * 0.1), currentMaxDiameter, blurred_image
        )
        uniqueCircles = merge_and_remove_duplicates(uniqueCircles, currentCircles)

    return uniqueCircles


def saveCircleROIsToBMP(circle_rois, subfolder_name, base_folder="detectedErrors"):
    """
    Saves each ROI in circle_rois as a BMP file in a specified subfolder within 'detectedErrors'.

    Args:
        circle_rois (list): A list of ROI images to save.
        subfolder_name (str): The name of the subfolder where the files will be saved.
        base_folder (str, optional): The base folder where the subfolder will be created. Defaults to 'detectedErrors'.

    Returns:
        None
    """
    # Normalize the path to handle any inconsistencies
    normalized_path = os.path.normpath(subfolder_name)
    # Extract the last folder name
    last_folder = os.path.basename(normalized_path)
    # Create the full path to the subfolder
    folder_path = os.path.join(base_folder, last_folder)

    # Creates the subfolder if it doesn’t already exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save each ROI as a BMP file
    for idx, roi in enumerate(circle_rois):
        filename = os.path.join(folder_path, f"circle_roi_{idx + 1}.bmp")
        cv.imwrite(filename, roi)

    print(f"All found Defects were stored in '{folder_path}' as .BMPs")


# draws circles for visual representation and uses saveCircleROIsToBMP to save the defects for later use
def checkAndDrawCircles(
    workImage,
    circles,
    minimalDiameter,
    maximalDiameter,
    maximalMeanIntensity,
    imageCopy,
    filename,
    drawcircles,
):
    circle_rois = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        errorCount = 0
        for i in circles[0, :]:
            center = (i[0], i[1])  # Center of Circle
            radius = i[2]  # Radius of Circle
            # Cutting area around Center
            circle_roi = workImage[
                center[1] - radius : center[1] + radius,
                center[0] - radius : center[0] + radius,
            ]
            circle_roi_save = imageCopy[
                center[1] - (radius + 20) : center[1] + (radius + 20),
                center[0] - (radius + 20) : center[0] + (radius + 20),
            ]
            # Calculating Mean
            mean_intensity = np.mean(circle_roi)
            # std_intensity = np.std(circle_roi)
            # check if Mean intensity and radius is under the threshold 
            if (
                mean_intensity < maximalMeanIntensity
                and minimalDiameter <= 2 * radius <= maximalDiameter
            ):
                errorCount += 1
                circle_rois.append(circle_roi_save)  # add ROI to List
                if drawcircles:
                    cv.circle(
                        imageCopy, center, radius + 5, (0, 255, 0), 4
                    )  # draws Circle on IMage for representation

    if not drawcircles:
        filemanagement.saveROIsToBMP(circle_rois, defectType= filemanagement.DefectType.WHISKERS, subfolder_name=filename)  # saves ROIs
    return errorCount


# print on console to check the parameters and how they perform.
# only use in combinaton with visual feedback. More found defects doesn´t automatically mean better performance.
def errorCountChecker(
    absolutMinimalDiameter,
    absolutMaximalDiameter,
    absolutMaximalMeanIntensity,
    numberOfErrorsDetected,
):
    print(
        "Number of Defects detected with \n\t minimal Diameter = ",
        absolutMinimalDiameter,
        "\n\t maximal Diameter = ",
        absolutMaximalDiameter,
        "\n\t maximal Mean Intensity = ",
        absolutMaximalMeanIntensity,
        "\n\t Defects =",
        numberOfErrorsDetected,
    )



def finishedSearchWhiskers(folderpath, show_Image):
    """
    use this when working directly from IFM-Folder

    filenameAndPath = filemanagement.find_largest_file(folderpath)
    # filenameAndPath = os.path.join(folderpath, f'probeOnly.bmp')
    filenameAndPath = imageProcessing(filenameAndPath, folderpath)
    
    image, blurred_image = loadPicture(filenameAndPath=filenameAndPath)
    """
    image, blurred_image = loadPicture(filenameAndPath=folderpath)


    # defining parameters
    absolutMinimalDiameter = 30
    absolutMaximalDiameter = 100
    absolutMaximalMeanIntensity = 180

    circles = findCircles(
        absolutMinimalDiameter, absolutMaximalDiameter, blurred_image
    )
    numberOfErrorsDetected = checkAndDrawCircles(
        workImage=blurred_image,
        circles=circles,
        minimalDiameter=absolutMinimalDiameter,
        maximalDiameter=absolutMaximalDiameter,
        maximalMeanIntensity=absolutMaximalMeanIntensity,
        imageCopy=image,
        filename=folderpath,
        drawcircles=show_Image,
    )
    errorCountChecker(
        absolutMinimalDiameter,
        absolutMaximalDiameter,
        absolutMaximalMeanIntensity,
        numberOfErrorsDetected,
    )

    # scales Picture for output
    width = 600
    height = int((width / image.shape[1]) * image.shape[0])  # scaling height to width
    resized_image = cv.resize(image, (width, height))

    # shows picture
    if show_Image:
        cv.imshow("Detected Defects", resized_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

