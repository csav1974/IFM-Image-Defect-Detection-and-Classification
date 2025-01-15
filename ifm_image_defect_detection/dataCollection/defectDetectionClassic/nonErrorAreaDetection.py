import cv2
import ifm_image_defect_detection.dataCollection.defectDetectionClassic.filemanagement as filemanagement
import tkinter as tk
from PIL import Image, ImageTk

# global Variable to stop program
stop_processing = False
selected_label = None

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

    # Resize the image by 500%
    new_size = (patch_pil.width * 5, patch_pil.height * 5)
    patch_pil_resized = patch_pil.resize(new_size, Image.NEAREST)

    # Convert resized image to Tkinter format
    patch_tk = ImageTk.PhotoImage(patch_pil_resized)

    # Display the image in the Tkinter window
    label = tk.Label(root, image=patch_tk)
    label.image = patch_tk  # Keep a reference to avoid garbage collection
    label.pack()

    # Hinweis-Label
    info_label = tk.Label(root, text="Drücke 'T' für True, 'F' für False, 'Q' zum Beenden")
    info_label.pack()

    # Tastatureingaben binden
    root.bind('t', lambda event: close_window(True))
    root.bind('T', lambda event: close_window(True))
    root.bind('f', lambda event: close_window(False))
    root.bind('F', lambda event: close_window(False))
    root.bind('q', lambda event: quit_processing)
    root.bind('Q', lambda event: quit_processing)

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
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    root.mainloop()


def nonErrorArea(filepath, folderpath):
    global stop_processing, selected_label

    image = cv2.imread(filepath)

    patch_size = 200
    stride = 100

    height, width, _ = image.shape

    rois = []
    # Divide the image into patches
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            if stop_processing:
                print("Processing stopped by user.")
                break

            patch = image[y : y + patch_size, x : x + patch_size]

            show_image(patch)

            if stop_processing:
                print("Processing stopped by user.")
                break

            if selected_label:
                rois.append(patch)
                print("Patch accepted")
            else:
                print("Patch rejected.")

    filemanagement.saveROIsToBMP(
        rois=rois,
        defectType=filemanagement.DefectType.NO_ERROR,
        subfolder_name=folderpath,
    )


# run the function
nonErrorArea(
    "sampleOnlyBMP/20241104_A2-1_3.bmp",
    "dataCollection/Data/detectedErrors/manualWhiskers20241104_A2-1/Whiskers",
)
