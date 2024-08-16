import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def count_defects(image_path, threshold_value=50, max_value=255, margin=5, min_area=0, max_area=500):
    # Laden des Bildes
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None, f"Das Bild konnte nicht geladen werden: {image_path}"
    
    # Bildgröße erhalten
    height, width = image.shape
    
    # Region of Interest (ROI) definieren: margin Pixel von jeder Seite
    roi = image[margin:height-margin, margin:width-margin]
    
    # Bild binarisieren: Defekte (schwarze Punkte) als weiße Punkte
    _, binary_image = cv2.threshold(roi, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    
    # Schließen (morphologische Operation) anwenden, um kleine Lücken zu schließen
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Finden von zusammenhängenden Komponenten
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtere Konturen nach Größe
    filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    
    # Zählen der Defekte
    num_defects = len(filtered_contours)
    
    # Berechnung der "dead area" (Verhältnis von schwarzen Pixeln zu weißen Pixeln)
    total_pixels = binary_image.size
    black_pixels = np.count_nonzero(binary_image)
    white_pixels = total_pixels - black_pixels
    dead_area_ratio = (black_pixels / total_pixels)*100 if total_pixels > 0 else 0
    
    # Markieren und Nummerieren der Defekte im Bild (optional)
    marked_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(filtered_contours):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"]) + margin
            centroid_y = int(M["m01"] / M["m00"]) + margin
            cv2.putText(marked_image, str(i + 1), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 1)
    
    return num_defects, dead_area_ratio, marked_image, None

def select_files_and_process():
    # Öffne Dateidialog zum Auswählen der PL-Aufnahmen
    file_paths = filedialog.askopenfilenames(title="Wähle PL-Aufnahmen aus", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    
    if not file_paths:
        print("Keine Dateien ausgewählt.")
        return
    
    # Datei zum Speichern der Ergebnisse
    results_file = os.path.join(os.path.dirname(file_paths[0]), "defect_count_results.txt")
    
    threshold_value = 50  # höherer Wert -> das Programm regstriert auch hellere Defekte 
    max_value = 255  # Normalerweise 255, kann jedoch geändert werden (grauwert)
    margin = 5  # Randbereich, der ignoriert wird (Falls das Bild nicht richtig zugeschnitten wurde)
    min_area = 0  # Minimale Größe eines Defekts
    max_area = 25  # Maximale Größe eines Defekts
    
    with open(results_file, "w") as f:
        for file_path in file_paths:
            num_defects, dead_area_ratio, marked_image, error = count_defects(file_path, threshold_value, max_value, margin, min_area, max_area)
            file_name = os.path.basename(file_path).replace('.png', '')
            if error:
                f.write(f"{file_name}: Fehler - {error}\n")
            else:
                f.write(f"{file_name}: Anzahl der Defekte - {num_defects}, Dead Area Ratio - {dead_area_ratio:.4f}\n")
                # Speichern des markierten Bildes (optional)
                marked_image_path = os.path.join(os.path.dirname(file_path), f"{file_name}_marked.png")
                cv2.imwrite(marked_image_path, marked_image)
    
    print(f"Analyse abgeschlossen. Ergebnisse wurden in {results_file} gespeichert.")

# Erstelle das Hauptfenster
root = tk.Tk()
root.withdraw()  # Verstecke das Hauptfenster

# Starte den Datei-Auswahl-Dialog und die Verarbeitung
select_files_and_process()
