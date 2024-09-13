import cv2
import dataCollection.filemanagement as filemanagement
import tkinter as tk
from PIL import Image, ImageTk

# global Variable to stop programm
stop_processing = False



def show_image(patch):
    global stop_processing, selected_label

    def close_window(is_no_Error):
        global selected_label
        selected_label = is_no_Error
        root.destroy()

    def quit_processing():
        global stop_processing
        stop_processing = True
        root.destroy()

    root = tk.Tk()
    root.title("Image Patch")

    # Convert OpenCV image to PIL image
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch_pil = Image.fromarray(patch_rgb)

    # Resize the image by 500% (5x)
    new_size = (patch_pil.width * 10, patch_pil.height * 10)
    patch_pil_resized = patch_pil.resize(new_size, Image.NEAREST)  # Using NEAREST to keep pixelation

    # Convert resized image to Tkinter format
    patch_tk = ImageTk.PhotoImage(patch_pil_resized)

    # Display the image in the Tkinter window
    label = tk.Label(root, image=patch_tk)
    label.image = patch_tk  # Keep a reference to avoid garbage collection
    label.pack()

    # Buttons for selection
    button_true = tk.Button(root, text="True", command=lambda: close_window(True))
    button_true.pack(side=tk.LEFT)
    
    button_false = tk.Button(root, text="False", command=lambda: close_window(False))
    button_false.pack(side=tk.RIGHT)
    
    # Quit Button
    button_quit = tk.Button(root, text="Quit", command=quit_processing)
    button_quit.pack(side=tk.BOTTOM)

    # Update the geometry to get the actual window size after packing
    root.update_idletasks()

    # Get the window size
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # Calculate the position to center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    # Set the window position
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    root.mainloop()



def nonErrorArea(filepath, folderpath):
    global stop_processing

    image = cv2.imread(filepath)
    
    patch_size = 32
    stride = 32

    height, width, _ = image.shape

    # Define the size of the square
    square_size = 900

    # Calculate the top-left corner of the square
    start_x = (width - square_size) // 2 
    start_y = (height - square_size) // 2 
    # Crop the square around the center
    image = image[start_y:start_y + square_size, start_x:start_x + square_size]

    height, width, _ = image.shape

    rois = []
    # Divide the image into patches
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):

            if stop_processing:
                print("Processing stopped by user.")
                break
            
            patch = image[y:y + patch_size, x:x + patch_size]
            
            show_image(patch)
        
            if stop_processing:
                print("Processing stopped by user.")
                break
            
            if selected_label:
                rois.append(patch)
            else:
                print("Patch rejected.")
            

    filemanagement.saveROIsToBMP(rois= rois, defectType= filemanagement.DefectType.NO_ERROR, subfolder_name= folderpath)




# run the function
nonErrorArea("sampleOnlyBMP/20240527_A9-2m$3D.bmp", "detectedErrors/20240527_A9-2m$3D")