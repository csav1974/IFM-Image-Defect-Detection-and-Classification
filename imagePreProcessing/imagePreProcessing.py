import os
import tkinter as tk
from tkinter import filedialog




def image_Processing_manual(filename, image_name="default", shape="circle"):
    if shape == "circle":
        from imagePreProcessing.processignCircular.processingCircularImage import bmpToProbeOnly_circle

        processed_image = bmpToProbeOnly_circle(filename)
    elif shape == "rectangle":
        from imagePreProcessing.processingRectangle.processingRectangularImage import bmpToProbeOnly_rectangle

        processed_image = bmpToProbeOnly_rectangle(filename)
    else:
        print("Invalid shape selected.")
        return None
    if processed_image is None:
        print("No image was saved")
        return None
    else:
        from imagePreProcessing.fileHandling.saveImage import saveBMPtoFolder

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

        # Get the parent folder name
        parent_folder = os.path.basename(os.path.dirname(filename))
        # Set the image_name_var to the parent folder name
        image_name_var.set(parent_folder)

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