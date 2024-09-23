import cv2 as cv
import numpy as np
import os



def bmpToProbeOnly_manual(filename, scale_factor=0.03):
    """
    Allows manual selection of a circular region in an image using a GUI.
    A scaled-down version of the image is used for the GUI to improve performance.
    The final image is saved in full resolution.

    Args:
        filename (str): Path to the input image file.
        scale_factor (float): Factor to scale down the image for the GUI (default is 0.25).
    """
    # Load the original full-resolution image
    image = cv.imread(filename)
    if image is None:
        print("Error loading image")
        return

    # Get dimensions of the original image
    orig_height, orig_width = image.shape[:2]

    # Scale down the image for the GUI
    scaled_width = int(orig_width * scale_factor)
    scaled_height = int(orig_height * scale_factor)
    scaled_image = cv.resize(image, (scaled_width, scaled_height), interpolation=cv.INTER_AREA)

    # Initial center and radius on the scaled image
    center_x = scaled_width // 2
    center_y = scaled_height // 2
    radius = min(center_x, center_y) - 10  # Initial radius

    # Create a window named 'Preview'
    cv.namedWindow('Preview')

    # Callback function to update the preview when trackbar values change
    def update_preview(x):
        # Get current positions of trackbars
        cx = cv.getTrackbarPos('Center X', 'Preview')
        cy = cv.getTrackbarPos('Center Y', 'Preview')
        r = cv.getTrackbarPos('Radius', 'Preview')

        # Create a mask with a filled circle at the specified center and radius
        mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
        cv.circle(mask, (cx, cy), r, 255, thickness=-1)

        # Apply the mask to the scaled image to get the circular cutout
        masked_image = cv.bitwise_and(scaled_image, scaled_image, mask=mask)

        # Display the preview image
        cv.imshow('Preview', masked_image)

    # Create trackbars for adjusting the center coordinates and radius
    cv.createTrackbar('Center X', 'Preview', center_x, scaled_width, update_preview)
    cv.createTrackbar('Center Y', 'Preview', center_y, scaled_height, update_preview)
    cv.createTrackbar('Radius', 'Preview', radius, min(scaled_width, scaled_height)//2, update_preview)

    # Initial call to display the image
    update_preview(0)

    print("Adjust the center and radius using the trackbars.")
    print("Press 's' to save the result, or 'q' to quit without saving.")

    while True:
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            # Get final positions of trackbars
            cx = cv.getTrackbarPos('Center X', 'Preview')
            cy = cv.getTrackbarPos('Center Y', 'Preview')
            r = cv.getTrackbarPos('Radius', 'Preview')

            # Map the center and radius back to the original image size
            scale_inv = 1 / scale_factor
            orig_cx = int(cx * scale_inv)
            orig_cy = int(cy * scale_inv)
            orig_r = int(r * scale_inv)

            # Create the final mask and masked image on the original image
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            cv.circle(mask, (orig_cx, orig_cy), orig_r, 255, thickness=-1)
            masked_image = cv.bitwise_and(image, image, mask=mask)

            # Create an alpha channel based on the mask
            b, g, r_img = cv.split(masked_image)
            alpha_channel = mask
            output_image = cv.merge((b, g, r_img, alpha_channel))

            # Close all OpenCV windows
            cv.destroyAllWindows()

            # Save the result
            return output_image
        elif key == ord('q'):
            print("Quitting without saving.")
            return None
            



# Transform an IFM Image to a blurred graylevel Image for further processing.
def TransformToBlurredGray(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ksize = (5, 5)  # Kernelgröße
    sigmaX = 1.0  # Standardabweichung in X-Richtung
    blurred_image = cv.GaussianBlur(gray, ksize, sigmaX)
    return blurred_image


# use HoughCircles to find only the probe itself.
# returns a list of circles.
def findCircles(minimalDiameter, maximalDiameter, image):
    circles = cv.HoughCircles(
        image,
        cv.HOUGH_GRADIENT,
        dp=3,
        minDist=3000,
        param1=150,
        param2=5,
        minRadius=minimalDiameter // 2,
        maxRadius=maximalDiameter // 2,
    )
    return circles


# takes the image and the blurred Version and matches it to the original image to create a image thats only the Probe with anything else black
def bmpToProbeOnly(filename):
    image = cv.imread(filename)
    blurredImage = TransformToBlurredGray(image)
    height, width, _ = image.shape
    min_diameter = int(min([height, width]) * 0.95)
    max_diameter = int(min([height, width]))
    circles = findCircles(
        minimalDiameter=min_diameter, maximalDiameter=max_diameter, image=blurredImage
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        center = (circles[0, 0][0], circles[0, 0][1])
        radius = circles[0, 0][2]

        # cv.circle(image, center, radius + 5, (0, 255, 0), 4)  # Umrandung des Kreises

        # cuts out the square of the Probe
        probeOnlyImage = image[
            center[1] - (radius) : center[1] + (radius),
            center[0] - (radius) : center[0] + (radius),
        ]

        # makes anything but the Probe black
        height, width = probeOnlyImage.shape[:2]
        center = (width // 2, height // 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.circle(mask, center, radius, (255), thickness=-1)
        masked_image = cv.bitwise_and(probeOnlyImage, probeOnlyImage, mask=mask)
        b, g, r = cv.split(masked_image)
        alpha_channel = mask
        return cv.merge((b, g, r, alpha_channel))


def bmpToSquare(filename):
    image = cv.imread(filename)
    height, width, _ = image.shape

    square_size = width // 2

    # Calculate the top-left corner of the square
    start_x = (width - square_size) // 2
    start_y = (height - square_size) // 2
    # Crop the square around the center
    image = image[start_y : start_y + square_size, start_x : start_x + square_size]

    return image


def saveBMPtoFolder(image, input_folder):
    """
    Saves a BMP image to a specified folder with a unique filename.
    If 'probeOnly_0.bmp' already exists, it appends a number to the filename.

    Args:
        image: The image to save.

    Returns:
        str: The full path to the saved file.
    """
    folder = "sampleOnlyBMP"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # create Filename based on the folder it comes from
    base_filename = os.path.basename(input_folder)
    extension = ".bmp"
    filename = os.path.join(folder, f"{base_filename}{extension}")

    print(f"saved to {filename}")
    # Save the image
    cv.imwrite(filename, image)

    return filename


def finishedProcessing(filename, folderpath):
    progressed_filename = saveBMPtoFolder(
        image=bmpToProbeOnly(filename), input_folder=folderpath
    )
    return progressed_filename

def image_Processing_manual(filename, folderpath):
    progressed_image = bmpToProbeOnly_manual(filename)
    if progressed_image is None:
        print("No Image was Safed")
        return None
    else:

        progressed_filename = saveBMPtoFolder(
            image=progressed_image, input_folder=folderpath
        )
        return progressed_filename
    

def dataCollectionSquare(filename, folderpath):
    progressed_filename = saveBMPtoFolder(
        image=bmpToSquare(filename), input_folder=folderpath
    )
    return progressed_filename


# # Bild auf gewünschte Größe skalieren
# width = 1000  # gewünschte Breite
# height = int((width / image.shape[1]) * image.shape[0])  # Höhe entsprechend skalieren
# resized_image = cv.resize(image, (width, height))

# cv.imshow('Detected Defects', resized_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

# dataCollectionSquare("sampleOnlyBMP/20240610_A6-2m_10x$3D.bmp", "sampleOnlyBMP")

image_Processing_manual("/home/georg/Work/UniversityJob/defectDetection/IFM_Data/2024/20240424$prj/20240424_A2-2m$3D_10x/texture.bmp", "/home/georg/Work/UniversityJob/defectDetection/IFM_Data/2024/20240424$prj/20240424_A2-2m$3D_10x")

