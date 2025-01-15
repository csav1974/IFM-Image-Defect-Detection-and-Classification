import cv2
import numpy as np
import tensorflow as tf
import os
import time
from ifm_image_defect_detection.csvHandling.writePredictionCSV import write_defect_data


def defect_recognition_old(image_path=None, model_name=None):
    # Load the model
    path_to_model = os.path.join("kerasModels", model_name)
    model = tf.keras.models.load_model(f"{path_to_model}.keras")
    IMG_SIZE = 128

    # Load and preprocess image
    image = cv2.imread(image_path)
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    patch_size = 128
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


def defect_recognition(image_path=None, model_name_1=None, model_name_2=None):
    """
    Recognize defects in an image using two neural network models.
    
    Parameters
    ----------
    image_path : str, optional
        The path to the input image to be processed.
    model_name_1 : str, optional
        The name of the first model used to classify patches as Error or No_Error.
    model_name_2 : str, optional
        The name of the second model used to further classify Error patches into 
        Whiskers, Chipping, or Scratches.
    
    Returns
    -------
    None
        The function writes the defect data to a CSV file using the `write_defect_data` function.
    """

    # Start measuring time for the entire script
    start_time_script = time.time()

    # Setting parameters
    IMG_SIZE = 128
    patch_size = 128
    stride = patch_size // 2

    ###############################################################################
    # 1) Load the first model
    ###############################################################################
    
    start_time = time.time()  # Capture start time for this block

    # Load the first model
    path_to_model_1 = os.path.join("kerasModels", model_name_1)
    model_1 = tf.keras.models.load_model(f"{path_to_model_1}.keras")
    
    end_time = time.time()  # Capture end time for this block
    print(f"Time for loading model 1: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 2) Load the second model
    ###############################################################################
    
    start_time = time.time()
    
    # Load the second model
    path_to_model_2 = os.path.join("kerasModels", model_name_2)
    model_2 = tf.keras.models.load_model(f"{path_to_model_2}.keras")
    
    end_time = time.time()
    print(f"Time for loading model 2: {end_time - start_time:.4f} seconds")



    ###############################################################################
    # 3) Read and preprocess the image
    ###############################################################################
    
    start_time = time.time()
    
    # Read the image from disk
    image = cv2.imread(image_path)
    # Convert image to grayscale
    work_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    end_time = time.time()
    print(f"Time for reading and preprocessing image: {end_time - start_time:.4f} seconds")

    height, width = work_image.shape

    ###############################################################################
    # 4) Convert the image to a Tensor
    ###############################################################################
    
    start_time = time.time()
    
    # Convert the grayscale image to a float Tensor
    work_tensor = tf.convert_to_tensor(work_image, dtype=tf.float32)
    # Add batch and channel dimensions
    work_tensor = tf.expand_dims(work_tensor, axis=0)
    
    work_tensor = tf.expand_dims(work_tensor, axis=-1)
    end_time = time.time()
    print(f"Time for converting to Tensor: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 5) Calculate the number of patches
    ###############################################################################

    start_time = time.time()
    
    # Calculate how many patches along each axis
    num_patches_y = ((height - patch_size) // stride) + 1
    num_patches_x = ((width - patch_size) // stride) + 1
    
    end_time = time.time()
    print(f"Time for calculating number of patches: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 6) Extract patches
    ###############################################################################

    start_time = time.time()

    # Extract patches from the Tensor
    patches_tf = tf.image.extract_patches(
        images=work_tensor,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    end_time = time.time()
    print(f"Time for extracting patches: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 7) Reshape and normalize patches
    ###############################################################################

    start_time = time.time()

    # Reshape to [num_patches_y, num_patches_x, patch_size, patch_size, 1]
    patches_tf = tf.reshape(patches_tf, [num_patches_y, num_patches_x, patch_size, patch_size, 1]) / 255.0

    end_time = time.time()
    print(f"Time for reshaping/normalizing patches: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 8) Create (x, y) coordinates
    ###############################################################################
    start_time = time.time()
    # Create (x, y) coordinates for each patch
    y_indices = tf.range(num_patches_y) * stride
    x_indices = tf.range(num_patches_x) * stride
    Y, X = tf.meshgrid(y_indices, x_indices, indexing='ij')
    Y = tf.reshape(Y, [-1])
    X = tf.reshape(X, [-1])
    end_time = time.time()
    print(f"Time for creating (x, y) coordinates: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 9) Flatten patches and resize to IMG_SIZE
    ###############################################################################

    start_time = time.time()

    # Flatten to [total_patches, patch_size, patch_size, 1]
    patches_tf = tf.reshape(patches_tf, [num_patches_y * num_patches_x, patch_size, patch_size, 1])
    
    # Resize patches to [IMG_SIZE, IMG_SIZE]
    patches_tf = tf.image.resize(patches_tf, [IMG_SIZE, IMG_SIZE], method='gaussian')

    end_time = time.time()
    print(f"Time for flattening/resizing patches: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 10) Convert Tensors to NumPy arrays
    ###############################################################################

    start_time = time.time()

    # Convert Tensors for patches and coordinates to NumPy
    patches_np = patches_tf.numpy()
    X_np = X.numpy()
    Y_np = Y.numpy()
    end_time = time.time()
    print(f"Time for converting to NumPy: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 11) Filter out patches that are entirely zero
    ###############################################################################
    start_time = time.time()
    # Only keep patches that are non-zero
    mask = np.all(patches_np != 0.0, axis=(1, 2, 3))
    patches_np = patches_np[mask]
    X_np = X_np[mask]
    Y_np = Y_np[mask]

    end_time = time.time()
    print(f"Time for filtering zero patches: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 12) Calculate batch size and steps
    ###############################################################################
    start_time = time.time()
    num_filtered_patches = patches_np.shape[0]
    batch_size = 32
    steps = int(np.ceil(num_filtered_patches / batch_size))
    end_time = time.time()
    print(f"Time for preparing batch size: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 13) Create a dataset for model_1
    ###############################################################################

    start_time = time.time()

    ds = tf.data.Dataset.from_tensor_slices(patches_np)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Predictions using model_1 (Error or No_Error)
    predictions_1 = model_1.predict(ds, verbose=1, steps=steps)
    # predictions_1 shape: (N, 2)
    # index 0 -> Error, index 1 -> No_Error
    predicted_classes_1 = np.argmax(predictions_1, axis=1)

    # Identify patches classified as Error
    error_mask = (predicted_classes_1 == 0)
    error_patches = patches_np[error_mask]
    error_X = X_np[error_mask]
    error_Y = Y_np[error_mask]

    end_time = time.time()
    print(f"Time for creating dataset for binary model: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 14) Apply model_2 only to the Error patches
    ###############################################################################

    start_time = time.time()

    if len(error_patches) > 0:
        steps_2 = int(np.ceil(error_patches.shape[0] / batch_size))
        ds_2 = tf.data.Dataset.from_tensor_slices(error_patches)
        ds_2 = ds_2.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        predictions_2 = model_2.predict(ds_2, verbose=1, steps=steps_2)
        # predictions_2 shape: (N_error, 3) -> [Whiskers, Chipping, Scratches]
    else:
        predictions_2 = np.empty((0, 3))

    end_time = time.time()
    print(f"Time for creating dataset for defect classification model: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 15) Construct final results
    ###############################################################################

    start_time = time.time()
    
    final_results = []
    error_idx = 0
    for i in range(num_filtered_patches):
        x_coord = X_np[i]
        y_coord = Y_np[i]

        if predicted_classes_1[i] == 1:
            # No_Error predicted by model_1 (index 1 -> No_Error)
            no_error_val = predictions_1[i, 1]
            # Whiskers, Chipping, Scratches = 0.0
            whiskers_val = 0.0
            chipping_val = 0.0
            scratches_val = 0.0
        else:
            # Error predicted by model_1 (index 0 -> Error)
            whiskers_val = predictions_2[error_idx, 0]
            chipping_val = predictions_2[error_idx, 1]
            scratches_val = predictions_2[error_idx, 2]
            # No_Error = 0 because model_1 indicated Error
            no_error_val = 0.0
            error_idx += 1

        final_results.append([
            int(x_coord),
            int(y_coord),
            [whiskers_val,
             chipping_val,
             scratches_val,
             no_error_val]
        ])

    end_time = time.time()
    print(f"Time for constructing results: {end_time - start_time:.4f} seconds")

    ###############################################################################
    # 16) Write results to a CSV file
    ###############################################################################

    start_time = time.time()

    write_defect_data(filename=image_path, patch_size=patch_size, stride=stride, data_list=final_results)

    end_time = time.time()
    print(f"Time for writing data to CSV: {end_time - start_time:.4f} seconds")

    # Measure total runtime
    end_time_script = time.time()
    print(f"Total script runtime: {end_time_script - start_time_script:.4f} seconds")

def main():
    defect_recognition(
        image_path="predictionDataCSV/Flexpecs_Cigs_T26/Flexpecs_Cigs_T26.bmp", 
        model_name_1="Model_20241229_firstStep",     # Binary model (No_Error / Error)
        model_name_2="Model_20241229_secondStep"     # Three-class model (Whiskers, Chipping, Scratches)
    )

if __name__ == "__main__":
    main()

