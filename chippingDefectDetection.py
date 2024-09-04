import cv2 as cv
import numpy as np
import filemanagement
from imagePreProcessing import finishedProcessing as imageProcessing

def chippingDetection(filepath, folderpath, show_Image):
    

    image = cv.imread(filepath)
    # Bild in Graustufen umwandeln
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Schwellenwert setzen, um helle Bereiche zu isolieren
    _, thresholded = cv.threshold(gray_image, 200, 255, cv.THRESH_BINARY)

    # Konturen finden
    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Liste für erkannte Defekte
    defects = []

    rois = []
    # Durch alle Konturen iterieren
    for contour in contours:
        # Begrenzendes Rechteck um die Kontur finden
        x, y, w, h = cv.boundingRect(contour)

        # Größe des Rechtecks überprüfen
        if 4 <= w <= 40 and 4 <= h <= 40:
            # Defekt gefunden, zur Liste hinzufügen
            defects.append((x, y, w, h))
            # Defekt auf dem Originalbild markieren
            if show_Image:
                cv.rectangle(image, (x, y), (x + w+5, y + h+5), (0, 255, 0), 2)
            
            #saving ROI
            roi_save = image[
                y - (h + 3) : y + (h + 3),
                x - (w + 3) : x + (w + 3),
            ]
            rois.append(roi_save)
    if not show_Image:
        filemanagement.saveROIsToBMP(rois= rois, defectType= filemanagement.DefectType.CHIPPING, subfolder_name= folderpath)
    # Anzahl gefundener Defekte
    print(f'Gefundene Defekte: {len(defects)}')

    # Bild mit markierten Defekten anzeigen
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
