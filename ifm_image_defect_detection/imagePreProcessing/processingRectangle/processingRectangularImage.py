import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ifm_image_defect_detection.imagePreProcessing.imageTransformation.rotateImage import rotate_image
from ifm_image_defect_detection.imagePreProcessing.fileHandling.loadImage import load_large_image

def bmpToProbeOnly_rectangle(filename, scale_factor=0.1):
    """
    Allows manual selection of a rectangular region in an image using a GUI.
    The image can be rotated in steps of 0.1 degrees.
    The rectangle remains horizontal (axis-aligned).
    The GUI window has a fixed size (75% of screen width) to prevent resizing during rotation.
    Sliders are placed below the image and stretch across the entire width.
    The final image is saved in full resolution.

    Args:
        filename (str): Path to the input image file.
        scale_factor (float): Factor to scale down the image for the GUI (default is 0.1 for 300MB Image).
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

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Selection")

    # Get screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Set window size to 75% of screen width
    window_width = int(screen_width * 0.75)
    # Maintain aspect ratio for height and add extra space for controls
    aspect_ratio = scaled_height / scaled_width
    control_height = 600  # Estimated height needed for controls
    if int(window_width * aspect_ratio) + control_height < screen_height:
        window_height = int(window_width * aspect_ratio) + control_height
    else: 
        window_width = int((screen_height - control_height) / aspect_ratio)
        window_height = screen_height


    # Set the window size
    root.geometry(f"{window_width}x{window_height}")
    # Prevent window from resizing
    root.resizable(False, False)

    # Create frames for image display and controls
    content_frame = tk.Frame(root)
    content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    image_frame = tk.Frame(content_frame)
    image_frame.pack(side=tk.TOP, fill=tk.BOTH)

    controls_frame = tk.Frame(content_frame)
    controls_frame.pack(side=tk.TOP, fill=tk.X)

    # Variables for sliders
    center_x_var = tk.IntVar(value=scaled_width // 2)
    center_y_var = tk.IntVar(value=scaled_height // 2)
    width_var = tk.IntVar(value=scaled_width * 0.9)
    height_var = tk.IntVar(value=scaled_height * 0.9)
    rotation_var = tk.IntVar(value=0)  # Rotation angle in tenths of degrees (initialized to 0 degrees)

    # Function to update the preview image
    def update_preview(*args):
        # Get current slider values
        cx = center_x_var.get()
        cy = center_y_var.get()
        w = width_var.get()
        h = height_var.get()
        rotation = rotation_var.get() / 10.0  # Convert to degrees

        # Rotate the scaled image
        rotated_scaled_image, _ = rotate_image(scaled_image, rotation)

        # Calculate translation due to rotation
        h_rot, w_rot = rotated_scaled_image.shape[:2]
        dx = (w_rot - scaled_width) // 2
        dy = (h_rot - scaled_height) // 2

        # Adjust center positions due to image size change
        cx_rot = cx + dx
        cy_rot = cy + dy

        # Create a mask with a filled rectangle at the specified center and size (not rotated)
        mask = np.zeros((h_rot, w_rot), dtype=np.uint8)
        top_left = (max(0, int(cx_rot - w // 2)), max(0, int(cy_rot - h // 2)))
        bottom_right = (min(w_rot, int(cx_rot + w // 2)), min(h_rot, int(cy_rot + h // 2)))
        cv2.rectangle(mask, top_left, bottom_right, 255, thickness=-1)

        # Apply the mask to the rotated scaled image to get the rectangular cutout
        masked_image = cv2.bitwise_and(rotated_scaled_image, rotated_scaled_image, mask=mask)

        # Convert the image to PIL format
        b, g, r = cv2.split(masked_image)
        img_rgb = cv2.merge((r, g, b))
        im_pil = Image.fromarray(img_rgb)

        # Resize image to fit in the window (subtract control height)
        im_pil = im_pil.resize((window_width, window_height - control_height), Image.LANCZOS)

        # Convert to ImageTk format
        imgtk = ImageTk.PhotoImage(image=im_pil)

        # Update the image in the label
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)

    # Function to save the image and exit
    def save_and_exit():
        # Get current slider values
        cx = center_x_var.get()
        cy = center_y_var.get()
        w = width_var.get()
        h = height_var.get()
        rotation = rotation_var.get() / 10.0  # Rotation angle in degrees

        # Map the center and size back to the original image size
        scale_inv = 1 / scale_factor
        orig_cx = cx * scale_inv
        orig_cy = cy * scale_inv
        orig_w = w * scale_inv
        orig_h = h * scale_inv

        # Rotate the original image
        rotated_image, _ = rotate_image(image, rotation)

        # Calculate translation due to rotation
        h_rot, w_rot = rotated_image.shape[:2]
        dx = (w_rot - orig_width) // 2
        dy = (h_rot - orig_height) // 2

        # Adjust center positions due to image size change
        orig_cx_rot = orig_cx + dx
        orig_cy_rot = orig_cy + dy

        # Create the rectangle in the rotated image
        top_left = (max(0, int(orig_cx_rot - orig_w // 2)), max(0, int(orig_cy_rot - orig_h // 2)))
        bottom_right = (min(w_rot, int(orig_cx_rot + orig_w // 2)), min(h_rot, int(orig_cy_rot + orig_h // 2)))

        # Crop the rectangle from the rotated image
        output_image = rotated_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Close the Tkinter window
        root.destroy()

        # Store the result in the root window
        root.output_image = output_image

    # Function to quit without saving
    def quit_without_saving():
        print("Quitting without saving.")
        root.destroy()
        root.output_image = None

    # Function to increment or decrement slider values
    def increment(delta, var, from_, to, command):
        value = var.get() + delta
        if value < from_:
            value = from_
        elif value > to:
            value = to
        var.set(value)
        command()

    # Function to create sliders with increment and decrement buttons
    def create_slider_with_buttons(frame, label, var, from_, to, command, step=1):
        slider_frame = tk.Frame(frame)
        slider_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        lbl = tk.Label(slider_frame, text=label)
        lbl.pack(side=tk.LEFT)

        btn_minus = tk.Button(slider_frame, text="-", command=lambda: increment(-step, var, from_, to, command))
        btn_minus.pack(side=tk.LEFT)

        scale = tk.Scale(slider_frame, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL, showvalue=False, command=lambda x: command())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_plus = tk.Button(slider_frame, text="+", command=lambda: increment(step, var, from_, to, command))
        btn_plus.pack(side=tk.LEFT)

        value_label = tk.Label(slider_frame, textvariable=var)
        value_label.pack(side=tk.LEFT)

    # Create a label to display the image
    image_label = tk.Label(image_frame)
    image_label.pack()

    # Create sliders
    create_slider_with_buttons(controls_frame, 'Center X', center_x_var, 0, scaled_width * 2, update_preview)
    create_slider_with_buttons(controls_frame, 'Center Y', center_y_var, 0, scaled_height * 2, update_preview)
    create_slider_with_buttons(controls_frame, 'Width', width_var, 1, scaled_width, update_preview)
    create_slider_with_buttons(controls_frame, 'Height', height_var, 1, scaled_height, update_preview)
    create_slider_with_buttons(controls_frame, 'Rotation', rotation_var, 0, 3600, update_preview)

    # Create Save and Quit buttons
    button_frame = tk.Frame(controls_frame)
    button_frame.pack(side=tk.TOP, pady=10)

    save_button = tk.Button(button_frame, text="Save", command=save_and_exit)
    save_button.pack(side=tk.LEFT, padx=5)

    quit_button = tk.Button(button_frame, text="Quit", command=quit_without_saving)
    quit_button.pack(side=tk.LEFT, padx=5)

    # Initial call to display the image
    update_preview()

    # Start the Tkinter main loop
    root.mainloop()

    # After the window is closed, return the output image
    return getattr(root, 'output_image', None)