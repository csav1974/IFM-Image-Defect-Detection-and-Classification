import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Pfad zum Eingabebild
image_path = 'testdata/20240829_A1-3_test.bmp'  # Passen Sie den Pfad zum Bild an

# Bild laden
image = cv2.imread(image_path)

# Überprüfen, ob das Bild geladen wurde
if image is None:
    print('Bild konnte nicht geladen werden. Bitte überprüfen Sie den Pfad.')
    exit()

# Konvertieren in Graustufen
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Rauschentfernung durch Gaussian Blur
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Kontrastverstärkung durch Histogrammgleichung
equalized = cv2.equalizeHist(gray_blurred)

# Gamma-Korrektur zur Verstärkung kleiner Helligkeitsunterschiede
gamma = 1.5  # Gamma-Wert anpassen (größer als 1 verstärkt Helligkeitsunterschiede)
look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
adjusted = cv2.LUT(equalized, look_up_table)

# Bildabmessungen
height, width = adjusted.shape

# Höhe eines Streifens (10% der Bildhöhe)
stripe_height = int(0.1 * height)

# Anzahl der Streifen (ohne Überlappung)
num_stripes = height // stripe_height

# Erstellen des Grids für die Subplots
fig = plt.figure(figsize=(16, 3 * num_stripes))  # Breitere Figur für größere Bilder
gs = gridspec.GridSpec(num_stripes, 2, width_ratios=[1, 1])  # Zwei Spalten

# Verarbeitung jedes Streifens
for i in range(num_stripes):
    # Bestimmen der y-Koordinaten des Streifens
    y_start = i * stripe_height
    y_end = y_start + stripe_height
    if y_end > height:
        y_end = height

    # Extrahieren des Streifens aus dem angepassten Bild
    stripe_adjusted = adjusted[y_start:y_end, :]

    # Berechnung der durchschnittlichen Helligkeit entlang der x-Koordinate
    brightness_profile = np.mean(stripe_adjusted, axis=0)

    # Kurvenglättung mittels gleitendem Durchschnitt
    window_size = 11  # Größe des Glättungsfensters anpassen (ungerade Zahl)
    brightness_profile_smooth = np.convolve(brightness_profile, np.ones(window_size)/window_size, mode='same')

    # Plotten des geglätteten Helligkeitsprofils
    ax_plot = fig.add_subplot(gs[i, 0])
    ax_plot.plot(brightness_profile_smooth, label='Geglättetes Profil')
    ax_plot.plot(brightness_profile, alpha=0.3, label='Originalprofil')  # Optional: Originalprofil anzeigen
    ax_plot.set_title(f'Streifen {i+1}: y = {y_start} bis {y_end}')
    ax_plot.set_xlabel('x-Koordinate')
    ax_plot.set_ylabel('Durchschnittliche Helligkeit')
    ax_plot.set_xlim([0, width - 1])
    ax_plot.set_ylim([0, 255])
    ax_plot.legend()

    # Extrahieren des entsprechenden Streifens aus dem Originalbild
    stripe_original = image[y_start:y_end, :]

    # Konvertieren für Matplotlib (BGR zu RGB)
    stripe_original_rgb = cv2.cvtColor(stripe_original, cv2.COLOR_BGR2RGB)

    # Anzeigen des Streifens
    ax_image = fig.add_subplot(gs[i, 1])
    ax_image.imshow(stripe_original_rgb)
    ax_image.set_title(f'Streifen {i+1} im Bild')
    ax_image.axis('off')

plt.tight_layout()
plt.show()
