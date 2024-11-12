import cv2 as cv
import dataCollection.defectDetectionClassic.filemanagement as filemanagement
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
        if 20 <= w <= 300 and 20 <= h <= 300:
            defects.append((x, y, w, h))
            # marks defect on image
            if show_Image:
                cv.rectangle(image, (x, y), (x + w + 20, y + h + 20), (0, 255, 0), 2)

            # saving ROI
            roi_save = image[
                y - (h + 40) : y + (h + 40),
                x - (w + 40) : x + (w + 40),
            ]
            rois.append(roi_save)
    if not show_Image:
        filemanagement.saveROIsToBMP(
            rois=rois, defectType=DefectType.CHIPPING, subfolder_name=folderpath
        )

    print(f"Number of found Chipping Defects: {len(defects)}")

    if show_Image:
        width = 600
        height = int(
            (width / image.shape[1]) * image.shape[0]
        )  # scaling height to width
        resized_image = cv.resize(image, (width, height))
        cv.imshow("Defekte", resized_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


####test
chippingDetection(
    filepath="sampleOnlyBMP/20240610_A6-2m_10x$3D_Square.bmp",
    folderpath="dataCollection/detectedErrors/20240610_A6-2m_10x$3D_Square",
    show_Image=False,
)

####
