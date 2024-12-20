import cv2
import numpy as np
import tensorflow as tf
import os
from csvHandling.writePredictionCSV import write_defect_data
from tqdm import tqdm

def defect_recognition(image_path = None, model_name = None):

    # Load Model. Assuming model is already trained
    path_to_model = os.path.join("kerasModels", model_name)
    model = tf.keras.models.load_model(f"{path_to_model}.keras")
    IMG_SIZE = 128  # scales the patch resolution down to IMG_SIZE*IMG_SIZE Pixel

    # Load the microscope image
    image = cv2.imread(image_path)
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define patch size
    patch_size = 128

    # Define stride (optional)
    stride = patch_size  // 2 # 50% overlap if stride = patch_size // 2

    # this part is used if only a part of the image should be examed for faster runtime. 
    # Only used for quick testing.
    # Comment out if you want to exam the whole picture
    # ############

    # start_x = 0
    # start_y = 0
    # square_size = 0

    # height, width = work_image.shape
    # # Define the size of the square
    # square_size = 2000

    # # Calculate the top-left corner of the square
    # start_x = (width - square_size) // 5
    # start_y = (height - square_size) // 5
    # # Crop the square around the center
    # image = image[start_y:start_y + square_size, start_x:start_x + square_size]
    # work_image = work_image[start_y:start_y + square_size, start_x:start_x + square_size]

    # ###########

    height, width = work_image.shape

    number_of_patches = len(range(0, height - patch_size, stride)) * len(
        range(0, width - patch_size, stride)
    )
    current_patch_number = 0

    data_list = []

    # Divide the image into patches
    for y in range(0, height - patch_size, stride):
        for x in range(0, width - patch_size, stride):
            patch = work_image[y : y + patch_size, x : x + patch_size]
            patch_resized = cv2.resize(
                patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
            )
            patch_resized = np.expand_dims(
                patch_resized / 255.0, axis=0
            )  # Normalize and add batch dimension

            # ignore black area around probe
            if np.any(patch_resized == 0):
                current_patch_number += 1
                continue
            # Classify the patch
            prediction = model.predict(patch_resized)

            row = [x, y, prediction[0]]
            data_list.append(row)

            current_patch_number += 1
            print(f"Patch: {current_patch_number} / {number_of_patches}")
            # progress_in_perzent = current_patch_number * 100 / number_of_patches
            # print(f"{progress_in_perzent:.2f}%")
            # print(prediction)

    write_defect_data(filename=image_path, patch_size=patch_size, stride=stride, data_list=data_list)


def defect_recognition_parallel_computing(image_path=None, model_name=None):
    # Load the model
    path_to_model = os.path.join("kerasModels", model_name)
    model = tf.keras.models.load_model(f"{path_to_model}.keras")
    IMG_SIZE = 128

    # Load and preprocess image
    image = cv2.imread(image_path)
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    patch_size = 256
    stride = patch_size // 2
    height, width = work_image.shape

    # Convert to Tensor
    work_tensor = tf.convert_to_tensor(work_image, dtype=tf.float32)
    # Shape: [height, width] -> [1, height, width, 1]
    work_tensor = tf.expand_dims(work_tensor, axis=0)
    work_tensor = tf.expand_dims(work_tensor, axis=-1)

    # Anzahl der Patches ermitteln
    num_patches_y = ((height - patch_size) // stride) + 1
    num_patches_x = ((width - patch_size) // stride) + 1

    # Extrahieren der Patches mit tf.image.extract_patches
    patches_tf = tf.image.extract_patches(
        images=work_tensor,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1,1,1,1],
        padding='VALID'
    )

    # Reshape in [num_patches, patch_size, patch_size]
    patches_tf = tf.reshape(patches_tf, [num_patches_y, num_patches_x, patch_size, patch_size])
    patches_tf = tf.expand_dims(patches_tf, axis=-1) 
    patches_tf = patches_tf / 255.0

    # Generiere (x,y)-Positionen für jedes Patch
    y_indices = tf.range(num_patches_y) * stride
    x_indices = tf.range(num_patches_x) * stride
    Y, X = tf.meshgrid(y_indices, x_indices, indexing='ij')
    Y = tf.reshape(Y, [-1])
    X = tf.reshape(X, [-1])
    
    # Flatten patches
    patches_tf = tf.reshape(patches_tf, [num_patches_y*num_patches_x, patch_size, patch_size, 1])

    # **Hier erfolgt die Größenänderung der Patches auf IMG_SIZE x IMG_SIZE:**
    patches_tf = tf.image.resize(patches_tf, [IMG_SIZE, IMG_SIZE], method='gaussian')

    # Um Anzahl der Schritte zu kennen, filtern wir jetzt auf Numpy-Ebene
    patches_np = patches_tf.numpy()  # zieht die Daten in den Host-Speicher (RAM)
    X_np = X.numpy()
    Y_np = Y.numpy()


    # Hier prüfen wir einfach: Gibt es irgendeinen Wert != 0?
    mask = np.any(patches_np != 0.0, axis=(1,2,3))
    patches_np = patches_np[mask]
    X_np = X_np[mask]
    Y_np = Y_np[mask]

    # Nun kennen wir die genaue Anzahl der Patches
    num_filtered_patches = patches_np.shape[0]
    batch_size = 16
    steps = int(np.ceil(num_filtered_patches / batch_size))

    # Dataset bauen
    ds = tf.data.Dataset.from_tensor_slices((patches_np, X_np, Y_np))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Vorhersage
    # Da Keras nun die Anzahl der Elemente kennt, kann es eine ordentliche Fortschrittsanzeige zeigen.
    predictions = model.predict(ds, verbose=1, steps=steps)

    # Koordinaten wieder aus ds extrahieren:
    # Da die Reihenfolge gleich ist, können wir die Koordinaten aus den Arrays X_np, Y_np nehmen
    # (Denn wir haben keinen Shuffle angewendet)
    data_list = []
    for (x_coord, y_coord, pred) in zip(X_np, Y_np, predictions):
        data_list.append([int(x_coord), int(y_coord), pred])

    # Write results to CSV
    write_defect_data(filename=image_path, patch_size=patch_size, stride=stride, data_list=data_list)

def main():
    defect_recognition_parallel_computing(
        image_path="predictionDataCSV/HZB_CIGS_4-4325-15-1-X$3D/HZB_CIGS_4-4325-15-1-X$3D.bmp", 
        model_name="Model_20241127"
    )
    
if __name__ == "__main__":
    main()

