import os
import cv2


def saveBMPtoFolder(image, image_name="default"):
    """
    Saves a BMP image to a specified folder with a unique filename.
    If a file with the same name exists, it appends a number to the filename.

    Args:
        image: The image to save.
        image_name (str): Name of the image file to be saved.

    Returns:
        str: The full path to the saved file.
    """

    # Assuming the script is always run from the project root directory (or a known directory within it)
    project_root = os.getcwd()

    # Specify the target folder within the project root
    folder = os.path.join(project_root, "sampleOnlyBMP")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create filename based on the provided image_name
    extension = ".bmp"
    filename = os.path.join(folder, f"{image_name}{extension}")

    # Ensure unique filename if file already exists
    count = 1
    base_filename = filename
    while os.path.exists(filename):
        filename = os.path.join(folder, f"{image_name}_{count}{extension}")
        count += 1

    # Save the image
    cv2.imwrite(filename, image)
    print(f"Saved to {filename}")

    return filename
