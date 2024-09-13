import os
import shutil

def copy_files_with_rename(source_dir, target_dir):
    # Erstelle den Zielordner, falls er nicht existiert
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Hole den Namen des Überordners des Quellordners
    parent_dir_name = os.path.basename(os.path.dirname(source_dir))

    # Durchlaufe alle Dateien im Quellordner
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)

        base, extension = os.path.splitext(filename)
        

        # Wenn die Datei bereits im Zielordner existiert, ändere den Namen
        if os.path.exists(target_file):
            counter = 1

            # Generiere einen neuen Dateinamen, der noch nicht existiert
            while os.path.exists(target_file):
                new_filename = f"{parent_dir_name}_{base}_{counter}{extension}"
                target_file = os.path.join(target_dir, new_filename)
                counter += 1
        else:
            new_filename = f"{parent_dir_name}_{base}{extension}"
            target_file = os.path.join(target_dir, new_filename)

        # Kopiere die Datei ins Zielverzeichnis
        shutil.copy2(source_file, target_file)
        print(f"'{filename}' wurde als '{os.path.basename(target_file)}' nach '{target_dir}' kopiert.")

# Beispielaufruf des Programms
source_directory_parent = "detectedErrors/20240527_A9-2m$3D/No_Error"  # Pfad zum Quellordner
target_directory_parent = "trainingdata/No_Error"   # Pfad zum Zielordner
'''
for defect in ["Chipping", "Whiskers", "No_Error"]: 
    source_directory = os.path.join(source_directory_parent, f"{defect}")
    target_directory = os.path.join(target_directory_parent, f"{defect}")
    copy_files_with_rename(source_directory, target_directory)
'''
copy_files_with_rename(source_directory_parent, target_directory_parent)