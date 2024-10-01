import numpy as np
import cv2
import os
import tkinter as tk
from csv_handling import read_from_csv
from dataCollection.savingDefcetsFromCSV import saveDefectsFromList
from enumDefectTypes import DefectType
from calculateDefectArea import calculate_defect_area_fromList

def main():
    csv_path = 'predictionDataCSV/unknown.csv'  # Path to the CSV file
    image_name, patch_size, stride, _, data_list = read_from_csv(csv_path)
    image_name = os.path.splitext(os.path.split(image_name)[-1])[0]
    # Read the image
    image_path = "sampleOnlyBMP/20240424_A2-2m$3D_10x.bmp"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load the image: {image_path}")
        return
    original_image = image.copy()  # Save the original image

    # Create a window
    window_name = 'Defect Visualization'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create trackbars (values from 0 to 1000, representing 0.0 to 1.0)
    cv2.createTrackbar('Threshold Whiskers', window_name, 0, 1000, lambda x: None)
    cv2.createTrackbar('Threshold Chipping', window_name, 0, 1000, lambda x: None)
    cv2.createTrackbar('Threshold Scratching', window_name, 0, 1000, lambda x: None)
    cv2.createTrackbar('Threshold No Defect', window_name, 0, 1000, lambda x: None)

    # Set default values for the trackbars
    cv2.setTrackbarPos('Threshold Whiskers', window_name, 900)
    cv2.setTrackbarPos('Threshold Chipping', window_name, 900)
    cv2.setTrackbarPos('Threshold Scratching', window_name, 900)
    cv2.setTrackbarPos('Threshold No Defect', window_name, 1000)

    # List to store rectangles and associated data
    rectangles = []

    # Variable for hover information
    hover_info = None

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal hover_info
        if event == cv2.EVENT_MOUSEMOVE:
            hover_info = None
            for rect in rectangles:
                x1, y1, x2, y2 = rect['coords']
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hover_info = {
                        'x': x1,
                        'y': y1,
                        'predictions': rect['predictions']
                    }
                    break

    # Set mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)

    # Define callback functions for buttons
    def save_image_callback(state,  event=None):
        save_path = 'output_image.bmp'
        cv2.imwrite(save_path, display_image)
        print(f"Image saved to {save_path}.")

    def save_whiskers_callback(state,  event=None):
        # Filter data_list for entries where prediction a > threshold_a
        threshold_a = cv2.getTrackbarPos('Threshold Whiskers', window_name) / 1000.0
        filtered_data_list = [(x, y, predictions) for x, y, predictions in data_list if predictions[0] > threshold_a]
        defect_type = DefectType.WHISKERS
        saveDefectsFromList(original_image, image_name, filtered_data_list, patch_size, defect_type)
        print(f"Saved ROIs for defect type: {defect_type}")

    def save_chipping_callback(state,  event=None):
        # Filter data_list for entries where prediction b > threshold_b
        threshold_b = cv2.getTrackbarPos('Threshold Chipping', window_name) / 1000.0
        filtered_data_list = [(x, y, predictions) for x, y, predictions in data_list if predictions[1] > threshold_b]
        defect_type = DefectType.CHIPPING
        saveDefectsFromList(original_image, image_name, filtered_data_list, patch_size, defect_type)
        print(f"Saved ROIs for defect type: {defect_type}")

    def save_scratching_callback(state,  event=None):
        # Filter data_list for entries where prediction c > threshold_c
        threshold_c = cv2.getTrackbarPos('Threshold Scratching', window_name) / 1000.0
        filtered_data_list = [(x, y, predictions) for x, y, predictions in data_list if predictions[2] > threshold_c]
        defect_type = DefectType.SCRATCHES
        saveDefectsFromList(original_image, image_name, filtered_data_list, patch_size, defect_type)
        print(f"Saved ROIs for defect type: {defect_type}")

    def save_no_defect_callback(state, event=None):
        # Filter data_list for entries where prediction d > threshold_d
        threshold_d = cv2.getTrackbarPos('Threshold No Defect', window_name) / 1000.0
        filtered_data_list = [(x, y, predictions) for x, y, predictions in data_list if predictions[3] > threshold_d]
        defect_type = DefectType.NO_ERROR
        saveDefectsFromList(original_image, image_name, filtered_data_list, patch_size, defect_type)
        print(f"Saved ROIs for defect type: {defect_type}")

    def show_defect_data_callback(state,  event=None):
        threshold_w = cv2.getTrackbarPos('Threshold Whiskers', window_name) / 1000.0
        threshold_c = cv2.getTrackbarPos('Threshold Chipping', window_name) / 1000.0
        threshold_s = cv2.getTrackbarPos('Threshold Scratching', window_name) / 1000.0
        data_list_with_defectType = []
        data_list_w = [(x, y, predictions) for x, y, predictions in data_list if predictions[0] > threshold_w]
        data_list_with_defectType.append([data_list_w, DefectType.WHISKERS])
        data_list_c = [(x, y, predictions) for x, y, predictions in data_list if predictions[1] > threshold_c]
        data_list_with_defectType.append([data_list_c, DefectType.CHIPPING])
        data_list_s = [(x, y, predictions) for x, y, predictions in data_list if predictions[2] > threshold_s]
        data_list_with_defectType.append([data_list_s, DefectType.SCRATCHES])

        data_list_with_defectType.append([[(x, y, predictions) for x, y, predictions in data_list if predictions[3] > 0.995], DefectType.NO_ERROR])
        
        # Calculate defect area
        defect_data = calculate_defect_area_fromList(image, data_list_with_defectType, patch_size)
        whiskers_area, chipping_area, scratches_area, defect_pixel, working_pixel, ratio = defect_data
        def create_data_window():

            # Create a tkinter window to display the results
            root = tk.Tk()
            root.title("Defect Data")

            # Add a frame to contain the labels for better formatting
            frame = tk.Frame(root, padx=20, pady=20)
            frame.pack()

            # Create labels with better formatting for each piece of information
            tk.Label(frame, text="Whiskers Area:", font=("Helvetica", 12), anchor="w").grid(row=0, column=0, sticky="w")
            tk.Label(frame, text=f"{whiskers_area}", font=("Helvetica", 12), anchor="e").grid(row=0, column=1, sticky="e")

            tk.Label(frame, text="Chipping Area:", font=("Helvetica", 12), anchor="w").grid(row=1, column=0, sticky="w")
            tk.Label(frame, text=f"{chipping_area}", font=("Helvetica", 12), anchor="e").grid(row=1, column=1, sticky="e")

            tk.Label(frame, text="Scratches Area:", font=("Helvetica", 12), anchor="w").grid(row=2, column=0, sticky="w")
            tk.Label(frame, text=f"{scratches_area}", font=("Helvetica", 12), anchor="e").grid(row=2, column=1, sticky="e")

            tk.Label(frame, text="Defect Pixels:", font=("Helvetica", 12), anchor="w").grid(row=3, column=0, sticky="w")
            tk.Label(frame, text=f"{defect_pixel}", font=("Helvetica", 12), anchor="e").grid(row=3, column=1, sticky="e")

            tk.Label(frame, text="Working Pixels:", font=("Helvetica", 12), anchor="w").grid(row=4, column=0, sticky="w")
            tk.Label(frame, text=f"{working_pixel}", font=("Helvetica", 12), anchor="e").grid(row=4, column=1, sticky="e")

            tk.Label(frame, text="Defect-to-Working Ratio:", font=("Helvetica", 12), anchor="w").grid(row=5, column=0, sticky="w")
            tk.Label(frame, text=f"{ratio:.2f}%", font=("Helvetica", 12), anchor="e").grid(row=5, column=1, sticky="e")

            # Start tkinter loop
            root.mainloop()
        create_data_window()
        
    # Create buttons
    cv2.createButton('Save Image', save_image_callback, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton('Save Whiskers ROIs', save_whiskers_callback, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton('Save Chipping ROIs', save_chipping_callback, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton('Save Scratching ROIs', save_scratching_callback, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton('Save No Defect ROIs', save_no_defect_callback, None, cv2.QT_PUSH_BUTTON, 0)
    cv2.createButton('Show Defect Data', show_defect_data_callback, None, cv2.QT_PUSH_BUTTON, 0)

    # Main loop
    while True:
        # Copy of the original image for updating
        display_image = original_image.copy()

        # Get current thresholds from the trackbars
        threshold_a = cv2.getTrackbarPos('Threshold Whiskers', window_name) / 1000.0
        threshold_b = cv2.getTrackbarPos('Threshold Chipping', window_name) / 1000.0
        threshold_c = cv2.getTrackbarPos('Threshold Scratching', window_name) / 1000.0
        threshold_d = cv2.getTrackbarPos('Threshold No Defect', window_name) / 1000.0

        # Clear the rectangle list
        rectangles.clear()

        # For each data point, check if thresholds are exceeded
        for entry in data_list:
            x, y, predictions = entry
            a, b, c, d = predictions

            # Check if any of the values exceed the threshold
            if (a > threshold_a):  # Threshold Whiskers
                # Draw rectangle on the image
                top_left = (x, y)
                bottom_right = (x + patch_size, y + patch_size)
                cv2.rectangle(display_image, top_left, bottom_right, (0, 0, 0), 2)
                # Save rectangle and associated data
                rectangles.append({
                    'coords': (x, y, x + patch_size, y + patch_size),
                    'predictions': predictions
                })
            if (b > threshold_b):  # Threshold Chipping
                # Draw rectangle on the image
                top_left = (x, y)
                bottom_right = (x + patch_size, y + patch_size)
                cv2.rectangle(display_image, top_left, bottom_right, (0, 0, 255), 2)
                # Save rectangle and associated data
                rectangles.append({
                    'coords': (x, y, x + patch_size, y + patch_size),
                    'predictions': predictions
                })
            if (c > threshold_c):  # Threshold Scratching
                # Draw rectangle on the image
                top_left = (x, y)
                bottom_right = (x + patch_size, y + patch_size)
                cv2.rectangle(display_image, top_left, bottom_right, (0, 255, 0), 2)
                # Save rectangle and associated data
                rectangles.append({
                    'coords': (x, y, x + patch_size, y + patch_size),
                    'predictions': predictions
                })
            if (d > threshold_d):  # Threshold No Defect
                # Draw rectangle on the image
                top_left = (x, y)
                bottom_right = (x + patch_size, y + patch_size)
                cv2.rectangle(display_image, top_left, bottom_right, (255, 255, 255), 2)
                # Save rectangle and associated data
                rectangles.append({
                    'coords': (x, y, x + patch_size, y + patch_size),
                    'predictions': predictions
                })

        # Display additional information when hovering
        if hover_info is not None:
            x1, y1 = hover_info['x'], hover_info['y']
            predictions = hover_info['predictions']
            # Create a semi-transparent overlay
            overlay = display_image.copy()
            # Define rectangle area
            cv2.rectangle(overlay, (x1, y1), (x1 + patch_size, y1 + patch_size), (0, 255, 0), -1)
            # Combine overlay with the image
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, display_image, 1 - alpha, 0, display_image)

            # Display predictions as text
            text = f"x: {x1}, y: {y1}, a: {predictions[0]:.3f}, b: {predictions[1]:.3f}, c: {predictions[2]:.3f}, d: {predictions[3]:.3f}"
            # Calculate position for the text
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + patch_size + 20
            # Draw background for the text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), (0, 0, 0), -1)
            # Put text on the image
            cv2.putText(display_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the image
        cv2.imshow(window_name, display_image)

        # Wait for user input
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC to exit
            break
        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
