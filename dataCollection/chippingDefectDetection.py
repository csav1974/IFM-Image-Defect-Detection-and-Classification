import cv2 as cv
import dataCollection.filemanagement as filemanagement
from dataCollection.imagePreProcessing import finishedProcessing as imageProcessing
from enumDefectTypes import DefectType

def chippingDetection(filepath, folderpath, show_Image):
    

    image = cv.imread(filepath)
    # convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # setting max brightnsess threshold
    _, thresholded = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    defects = []

    rois = []
    for contour in contours:
        # rectangle arounf defect
        x, y, w, h = cv.boundingRect(contour)

        # check for size
        if 4 <= w <= 80 and 4 <= h <= 80:

            defects.append((x, y, w, h))
            # marks defect on image
            if show_Image:
                cv.rectangle(image, (x, y), (x + w+5, y + h+5), (0, 255, 0), 2)
            
            #saving ROI
            roi_save = image[
                y - (h + 3) : y + (h + 3),
                x - (w + 3) : x + (w + 3),
            ]
            rois.append(roi_save)
    if not show_Image:
        filemanagement.saveROIsToBMP(rois= rois, defectType= DefectType.CHIPPING, subfolder_name= folderpath)
    
    print(f'Number of found Chipping Defects: {len(defects)}')

    
    if show_Image:
        width = 600
        height = int((width / image.shape[1]) * image.shape[0])  # scaling height to width
        resized_image = cv.resize(image, (width, height))
        cv.imshow('Defekte', resized_image)
        cv.waitKey(0)
        cv.destroyAllWindows()




def finishedSearchChipping(folderpath, show_Image):
    filenameAndPath = filemanagement.find_largest_file(folderpath)
    filenameAndPath = imageProcessing(filenameAndPath, folderpath)
    chippingDetection(filenameAndPath, folderpath, show_Image)
