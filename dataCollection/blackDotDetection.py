import cv2
import numpy as np
import csv
import os


# Read the microscope image
# Replace 'microscope_image.jpg' with your image file path
image_path = "sampleOnlyBMP/20240610_A6-2m_10x$3D.bmp"

image = cv2.imread(image_path)

height, width, _ = image.shape
# Define the size of the square
square_size = 4000

# Calculate the top-left corner of the square
start_x = (width - square_size) // 3
start_y = (height - square_size) // 3
# Crop the square around the center
image = image[start_y : start_y + square_size, start_x : start_x + square_size]

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve thresholding
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply inverse binary thresholding to get dark spots
# Adjust the threshold value if necessary
_, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to store defect data
defects = []

# Loop over the contours
for cnt in contours:
    # Calculate area of the contour
    area = cv2.contourArea(cnt)
    # Ignore small or large contours that are unlikely to be defects
    if area < 200 or area > 10000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    if (gray[y : y + height, x : x + width] > 220).sum() >= 4:
        continue
    # Calculate the equivalent diameter
    equi_diameter = np.sqrt(4 * area / np.pi)

    # Approximate the contour to a circle and compute circularity
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Consider contours that are roughly circular
    if circularity > 0.4:
        # Calculate centroid of the defect
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Add the defect information to the list
        defects.append((x, y, max([w, h])))


image_name = os.path.basename(image_path)  # Extract image name from path
image_size = square_size  # The size of the cropped square

# Create the CSV structure
csv_data = []

# First line: start_x, start_y, image_size, image_name
csv_data.append([start_x, start_y, image_size, image_name])

# Second line: Skip this (for future use)
csv_data.append(["Values skipped"])

# Third line: Add headers for the defect coordinates
csv_data.append(["x", "y", "patch_size"])

# Fourth and subsequent lines: Coordinates of defects
csv_data.extend(defects)

# Write to CSV file
csv_path = "defectPositionCSV/blackDotDefects.csv"
with open(csv_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)


# Optionally, draw the detected defects on the image and display it
for defect in defects:
    cv2.rectangle(
        image,
        (defect[0], defect[1]),
        (defect[0] + defect[2], defect[1] + defect[2]),
        (0, 0, 0),
        2,
    )

# Resize the image
resized_image = cv2.resize(image, (800, 800))

# Display the resized image with detected defects
cv2.imshow("Detected Defects", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
