import os
import shutil

def copy_files_with_rename(source_dir, target_dir):
    project_root = os.getcwd()
    source_dir = os.path.join(project_root, source_dir)
    target_dir = os.path.join(project_root, target_dir)
    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get the name of the parent directory of the source directory
    parent_dir_name = os.path.basename(os.path.dirname(source_dir))

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)

        base, extension = os.path.splitext(filename)

        # If the file already exists in the target directory, change the name
        if os.path.exists(target_file):
            counter = 1

            # Generate a new filename that does not yet exist
            while os.path.exists(target_file):
                new_filename = f"{parent_dir_name}_{base}_{counter}{extension}"
                target_file = os.path.join(target_dir, new_filename)
                counter += 1
        else:
            new_filename = f"{parent_dir_name}_{base}{extension}"
            target_file = os.path.join(target_dir, new_filename)

        # Copy the file to the target directory
        shutil.copy2(source_file, target_file)
        print(
            f"'{filename}' was copied as '{os.path.basename(target_file)}' to '{target_dir}'."
        )


def get_direct_subfolders(folder_path):
    # Prüfe, ob der übergebene Pfad ein gültiger Ordner ist
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} ist kein gültiger Ordnerpfad!")
    
    # Liste für die Ergebnisse
    subfolder_paths = []
    
    # Iteriere durch die Inhalte des Ordners
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        # Füge nur Ordner (keine Dateien) zur Liste hinzu
        if os.path.isdir(entry_path):
            subfolder_paths.append(entry_path)
    
    return subfolder_paths

# Beispielaufruf des Programms
# source_directory_parent = "dataCollection/Data/detectedErrors/machinefoundErrors/20240829_A1-1"  # Pfad zum Quellordner
# target_directory_parent = (
#     "dataCollection/Data/Training20240829_v3"  # Pfad zum Zielordner
# )
def copy_folder(source_directory_parent,target_directory_parent): 
    for defect in ["Chipping", "Whiskers", "Scratches", "No_Error"]: 
        source_directory = os.path.join(source_directory_parent, f"{defect}")
        target_directory = os.path.join(target_directory_parent, f"{defect}")
        copy_files_with_rename(source_directory, target_directory)

# copy_files_with_rename("dataCollection/Data/detectedErrors/machinefoundErrors/20240829_A1-2/Whiskers", "dataCollection/Data/Perfect_Data/20240829/Whiskers")


def main():
    print("imageCopy Main running...")
    source_folder = "dataCollection/Data/Perfect_Data"
    target_folder = "dataCollection/Data/TrainingData_2024_11_27"
    subfolders = get_direct_subfolders(source_folder)
    for subfolder in subfolders:
        print(subfolder)
        print (target_folder)
        copy_folder(subfolder, target_folder)

if __name__ == "__main__":
    main()
