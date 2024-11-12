import cv2
import numpy as np
from ..fileHandling.loadImage import load_large_image



def bmpToProbeOnly_circle(filename, scale_factor=0.03):
    """
    Allows manual selection of a circular region in an image using a GUI.
    A scaled-down version of the image is used for the GUI to improve performance.
    The final image is saved in full resolution.

    Args:
        filename (str): Path to the input image file.
        scale_factor (float): Factor to scale down the image for the GUI (default is 0.03 for a 900MB File).
    
    Return: 
        output_image (np.ndarray): edited image ready to save.
    """
    # Load the original full-resolution image
    image = load_large_image(filename)
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

        # Verify the radius is non-negative and within bounds
        if r < 0:
            r = 0

        # Create a mask only if the radius is valid
        if r > 0 and 0 <= cx < scaled_width and 0 <= cy < scaled_height:
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
