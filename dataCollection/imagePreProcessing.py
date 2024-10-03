import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def bmpToProbeOnly_circle(filename, scale_factor=0.03):
    """
    Allows manual selection of a circular region in an image using a GUI.
    A scaled-down version of the image is used for the GUI to improve performance.
    The final image is saved in full resolution.

    Args:
        filename (str): Path to the input image file.
        scale_factor (float): Factor to scale down the image for the GUI (default is 0.03).
    """
    # Load the original full-resolution image
    image = cv2.imread(filename)
    if image is None:
        print("Error loading image")
        return

    # Get dimensions of the original image
    orig_height, orig_width = image.shape[:2]

    # Scale down the image for the GUI
    scaled_width = int(orig_width * scale_factor)
    scaled_height = int(orig_height * scale_factor)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # Initial center and radius on the scaled image
    center_x = scaled_width // 2
    center_y = scaled_height // 2
    radius = min(center_x, center_y) - 10  # Initial radius

    # Create a window named 'Preview'
    cv2.namedWindow('Preview')

    # Callback function to update the preview when trackbar values change
    def update_preview(x):
        # Get current positions of trackbars
        cx = cv2.getTrackbarPos('Center X', 'Preview')
        cy = cv2.getTrackbarPos('Center Y', 'Preview')
        r = cv2.getTrackbarPos('Radius', 'Preview')

        # Create a mask with a filled circle at the specified center and radius
        mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

        # Apply the mask to the scaled image to get the circular cutout
        masked_image = cv2.bitwise_and(scaled_image, scaled_image, mask=mask)

        # Display the preview image
        cv2.imshow('Preview', masked_image)

    # Create trackbars for adjusting the center coordinates and radius
    cv2.createTrackbar('Center X', 'Preview', center_x, scaled_width, update_preview)
    cv2.createTrackbar('Center Y', 'Preview', center_y, scaled_height, update_preview)
    cv2.createTrackbar('Radius', 'Preview', radius, min(scaled_width, scaled_height)//2, update_preview)

    # Initial call to display the image
    update_preview(0)

    print("Adjust the center and radius using the trackbars.")
    print("Press 's' to save the result, or 'q' to quit without saving.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Get final positions of trackbars
            cx = cv2.getTrackbarPos('Center X', 'Preview')
            cy = cv2.getTrackbarPos('Center Y', 'Preview')
            r = cv2.getTrackbarPos('Radius', 'Preview')

            # Map the center and radius back to the original image size
            scale_inv = 1 / scale_factor
            orig_cx = int(cx * scale_inv)
            orig_cy = int(cy * scale_inv)
            orig_r = int(r * scale_inv)

            # Create the final mask and masked image on the original image
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            cv2.circle(mask, (orig_cx, orig_cy), orig_r, 255, thickness=-1)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Create an alpha channel based on the mask
            b, g, r_img = cv2.split(masked_image)
            alpha_channel = mask
            merged_image = cv2.merge((b, g, r_img, alpha_channel))
            if orig_cx + orig_r <= orig_width and orig_cy + orig_r <= orig_height:
                output_image = merged_image[orig_cy - orig_r : orig_cy + orig_r, orig_cx - orig_r : orig_cx + orig_r]
            else: 
                output_image = merged_image
            # Close all OpenCV windows
            cv2.destroyAllWindows()

            # Save the result
            return output_image
        elif key == ord('q'):
            print("Quitting without saving.")
            cv2.destroyAllWindows()
            return None

def bmpToProbeOnly_rectangle(filename, scale_factor=0.03):
    """
    Allows manual selection of a rectangular region in an image using a GUI.
    A scaled-down version of the image is used for the GUI to improve performance.
    The final image is saved in full resolution.

    Args:
        filename (str): Path to the input image file.
        scale_factor (float): Factor to scale down the image for the GUI (default is 0.03).
    """
    # Load the original full-resolution image
    image = cv2.imread(filename)
    if image is None:
        print("Error loading image")
        return

    # Get dimensions of the original image
    orig_height, orig_width = image.shape[:2]

    # Scale down the image for the GUI
    scaled_width = int(orig_width * scale_factor)
    scaled_height = int(orig_height * scale_factor)
    scaled_image = cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    # Initial center and size on the scaled image
    center_x = scaled_width // 2
    center_y = scaled_height // 2
    width = scaled_width // 2
    height = scaled_height // 2

    # Create a window named 'Preview'
    cv2.namedWindow('Preview')

    # Callback function to update the preview when trackbar values change
    def update_preview(x):
        # Get current positions of trackbars
        cx = cv2.getTrackbarPos('Center X', 'Preview')
        cy = cv2.getTrackbarPos('Center Y', 'Preview')
        w = cv2.getTrackbarPos('Width', 'Preview')
        h = cv2.getTrackbarPos('Height', 'Preview')

        # Create a mask with a filled rectangle at the specified center and size
        mask = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
        top_left = (max(0, cx - w // 2), max(0, cy - h // 2))
        bottom_right = (min(scaled_width, cx + w // 2), min(scaled_height, cy + h // 2))
        cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)

        # Apply the mask to the scaled image to get the rectangular cutout
        masked_image = cv2.bitwise_and(scaled_image, scaled_image, mask=mask)

        # Display the preview image
        cv2.imshow('Preview', masked_image)

    # Create trackbars for adjusting the center coordinates and size
    cv2.createTrackbar('Center X', 'Preview', center_x, scaled_width, update_preview)
    cv2.createTrackbar('Center Y', 'Preview', center_y, scaled_height, update_preview)
    cv2.createTrackbar('Width', 'Preview', width, scaled_width, update_preview)
    cv2.createTrackbar('Height', 'Preview', height, scaled_height, update_preview)

    # Initial call to display the image
    update_preview(0)

    print("Adjust the center, width, and height using the trackbars.")
    print("Press 's' to save the result, or 'q' to quit without saving.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Get final positions of trackbars
            cx = cv2.getTrackbarPos('Center X', 'Preview')
            cy = cv2.getTrackbarPos('Center Y', 'Preview')
            w = cv2.getTrackbarPos('Width', 'Preview')
            h = cv2.getTrackbarPos('Height', 'Preview')

            # Map the center and size back to the original image size
            scale_inv = 1 / scale_factor
            orig_cx = int(cx * scale_inv)
            orig_cy = int(cy * scale_inv)
            orig_w = int(w * scale_inv)
            orig_h = int(h * scale_inv)

            # Create the final mask and masked image on the original image
            mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            top_left = (max(0, orig_cx - orig_w // 2), max(0, orig_cy - orig_h // 2))
            bottom_right = (min(orig_width, orig_cx + orig_w // 2), min(orig_height, orig_cy + orig_h // 2))
            cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Create an alpha channel based on the mask
            b, g, r_img = cv2.split(masked_image)
            alpha_channel = mask
            if orig_cy - (orig_h // 2) >= 0 and orig_cy + (orig_h // 2) <= orig_height and orig_cx - (orig_w // 2) >= 0 and orig_cx + (orig_w // 2) <= orig_width:
                output_image = image[orig_cy - (orig_h // 2) : orig_cy + (orig_h // 2), orig_cx - (orig_w // 2) : orig_cx + (orig_w // 2)]
            else: 
                output_image = cv2.merge((b, g, r_img, alpha_channel))
            # Close all OpenCV windows
            cv2.destroyAllWindows()

            # Save the result
            return output_image
        elif key == ord('q'):
            print("Quitting without saving.")
            cv2.destroyAllWindows()
            return None

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
    folder = "sampleOnlyBMP"
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

    print(f"Saved to {filename}")
    # Save the image
    cv2.imwrite(filename, image)

    return filename

def image_Processing_manual(filename, image_name="default", shape="circle"):
    if shape == "circle":
        processed_image = bmpToProbeOnly_circle(filename)
    elif shape == "rectangle":
        processed_image = bmpToProbeOnly_rectangle(filename)
    else:
        print("Invalid shape selected.")
        return None
    if processed_image is None:
        print("No image was saved")
        return None
    else:
        processed_filename = saveBMPtoFolder(
            image=processed_image, image_name=image_name
        )
        return processed_filename
def main():
    # Initialize the main window
    root = tk.Tk()
    root.title("Image Selector")

    # Variables to store the selected image, shape, and image name
    selected_image = tk.StringVar()
    shape_var = tk.StringVar(value="circle")  # default shape is circle
    image_name_var = tk.StringVar()  # To store the name of the image

    # Function to select image
    def select_image():
        filename = filedialog.askopenfilename(title="Select Image", 
                                              filetypes=[("All files", "*.*")])
        selected_image.set(filename)
        print(f"Selected image: {filename}")

    # Function to start editing
    def edit_image():
        filename = selected_image.get()
        shape = shape_var.get()
        image_name = image_name_var.get()

        if not filename:
            print("No image selected.")
            return

        if not image_name:
            print("No image name provided.")
            return

        # Close the Tkinter window
        root.destroy()
        # Call image processing function
        image_Processing_manual(filename, image_name=image_name, shape=shape)

    # Create widgets
    select_button = tk.Button(root, text="Select Image", command=select_image)
    select_button.pack(pady=10)

    # Shape selection radio buttons
    shape_label = tk.Label(root, text="Select Shape:")
    shape_label.pack()
    circle_radio = tk.Radiobutton(root, text="Circle", variable=shape_var, value="circle")
    circle_radio.pack()
    rectangle_radio = tk.Radiobutton(root, text="Rectangle", variable=shape_var, value="rectangle")
    rectangle_radio.pack()

    # Image Name entry field
    image_name_label = tk.Label(root, text="Enter Image Name:")
    image_name_label.pack(pady=10)
    image_name_entry = tk.Entry(root, textvariable=image_name_var)
    image_name_entry.pack(pady=5)

    # Edit Image button
    edit_button = tk.Button(root, text="Edit Image", command=edit_image)
    edit_button.pack(pady=10)

    root.update_idletasks()

    # Calculate the position to center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = root.winfo_width() + screen_width // 4
    window_height = root.winfo_height() + screen_height // 4

    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    # Set the window position
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.mainloop()

if __name__ == "__main__":
    main()
