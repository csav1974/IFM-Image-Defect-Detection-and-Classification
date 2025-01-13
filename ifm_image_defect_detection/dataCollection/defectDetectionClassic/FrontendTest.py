import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from threading import Thread
from dataCollection.WiskersDetection import finishedSearchWhiskers
from dataCollection.chippingDefectDetection import finishedSearchChipping


# Main application window
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing")

        # Layout for folder selection
        self.folder_frame = ttk.Frame(root, padding="10")
        self.folder_frame.grid(row=0, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.folder_label = ttk.Label(self.folder_frame, text="No folder selected")
        self.folder_label.grid(row=0, column=0, padx=5, pady=5)

        self.select_button = ttk.Button(
            self.folder_frame, text="Select Folder", command=self.select_folder
        )
        self.select_button.grid(row=1, column=0, padx=5, pady=5)

        self.run_button = ttk.Button(
            self.folder_frame, text="Run", command=self.start_processing
        )
        self.run_button.grid(row=2, column=0, padx=5, pady=5)

        # Quit button
        self.quit_button = ttk.Button(self.folder_frame, text="Quit", command=root.quit)
        self.quit_button.grid(row=3, column=0, padx=5, pady=5)

        # Label and progress bar for loading indicator
        self.loading_frame = ttk.Frame(root, padding="10")
        self.loading_frame.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E))

        self.loading_label = ttk.Label(self.loading_frame, text="")
        self.loading_label.grid(row=0, column=0, padx=5, pady=5)

        self.progress = ttk.Progressbar(self.loading_frame, mode="indeterminate")
        self.progress.grid(row=1, column=0, padx=5, pady=5)

        self.selected_folder = None

    def select_folder(self):
        """Opens a file dialog to select a folder and updates the label with the selected path."""
        folder_path = filedialog.askdirectory(title="Please select a folder")

        if folder_path:
            self.selected_folder = os.path.normpath(folder_path)
            self.folder_label.config(text=f"Selected Folder: {self.selected_folder}")

    def start_processing(self):
        """Starts the processing in a new thread and shows the loading indicator."""
        if not self.selected_folder:
            messagebox.showwarning("Select Folder", "Please select a folder first.")
            return

        # Show loading indicator
        self.loading_label.config(text="Processing, please wait...")
        self.progress.start()

        # Run the processing in a separate thread
        thread = Thread(target=self.run_processing)
        thread.start()

    def run_processing(self):
        """Runs the image processing and hides the loading indicator."""
        try:
            finishedSearchWhiskers(
                self.selected_folder, show_Image=False
            )  # drawcircles=True shows found defects, drawcircles False saves all found defekts

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.progress.stop()
            self.loading_label.config(text="")


# Main function to start the GUI
def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
