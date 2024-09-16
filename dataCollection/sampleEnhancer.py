import os
import cv2


# use to mirror and rotate images to enlargen samplesize for training models
# takes the folder with actual images and creates a rotated and mirrored version
# do not use more then once per folder!
def artificial_Sample_enhance(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    # List all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            # Read the image
            image = cv2.imread(file_path)

            # Rotate the image by 90 degrees
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            # Mirror the image vertically
            mirrored_image = cv2.flip(image, 0)

            # Create new file paths for the rotated and mirrored images
            rotated_file_path = os.path.join(folder_path, f"rotated_{filename}")
            mirrored_image_path = os.path.join(folder_path, f"mirrored_{filename}")

            # Save the rotated and mirrored images
            cv2.imwrite(rotated_file_path, rotated_image)
            cv2.imwrite(mirrored_image_path, mirrored_image)


artificial_Sample_enhance(
    "dataCollection/detectedErrors/machinefoundErrors/20240610_A6-2m_10x$3D_Square/Chipping"
)
