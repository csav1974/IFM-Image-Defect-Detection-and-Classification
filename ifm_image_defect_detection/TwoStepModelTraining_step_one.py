import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def train_and_save_model_binary(DATADIR = "dataCollection/Data/Perfect_Data", model_name = "Model_20241229_binary.keras"):
    """
    Trains a binary classification model for defect detection.

    This function:
    - Loads and preprocesses image data from the specified directory.
    - Assigns binary labels to images based on the defined defect categories:
      * Defects (e.g., "Whiskers", "Chipping", "Scratching") are labeled as 0.
      * Non-defects (e.g., "Unknown_Defects", "No_Error") are labeled as 1.
    - Normalizes and reshapes the image data for training.
    - Splits the data into training and validation datasets.
    - Applies data augmentation to the training data.
    - Defines and compiles a Convolutional Neural Network (CNN) model.
    - Trains the model using the training data, with validation monitoring.
    - Saves the best-performing model to the specified path.
    - Prints the model's summary and dataset statistics.

    Parameters:
    - DATADIR (str): Path to the directory containing the categorized image data.
    - model_name (str): Name of the file to save the trained model.

    Returns:
    None
    """

    path_to_model = os.path.join("kerasModels", model_name)

    # Categories of defects
    CATEGORIES = [
        "Whiskers", 
        "Chipping", 
        "Scratching", 
        "Unknown_Defects",
        "No_Error",
    ] 

    IMG_SIZE = 128
    batch_size = 16

    def list_subfolders(folder_path):
        """
        Returns a list of full paths to all subfolders within a given folder_path.
        """
        subfolders = []
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):
                subfolders.append(full_path)
        return subfolders

    def create_training_data():
        """
        Reads images from each category, converts them to grayscale,
        resizes them to IMG_SIZE, and assigns a class label:
        - 0 for 'Whiskers', 'Chipping', 'Scratching'
        - 1 for 'Unknown_Defects', 'No_Error'
        """
        folderpaths = list_subfolders(DATADIR)
        training_data = []
        
        for folderpath in folderpaths:
            for category in CATEGORIES:
                if category in ["Whiskers", "Chipping", "Scratching"]:
                    class_num = 0
                else:
                    class_num = 1

                path = os.path.join(folderpath, category)
                if not os.path.isdir(path):
                    # Skip if the category folder doesn't exist
                    continue

                for img in tqdm(os.listdir(path), desc=f"Processing {category}"):
                    try:
                        # Read image in grayscale
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                        # Resize image
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                        training_data.append([new_array, class_num])
                    except Exception:
                        # Ignore any broken images or read errors
                        pass
        return training_data

    # Create training data
    training_data = create_training_data()
    random.shuffle(training_data)

    # Split features (X) and labels (y)
    X, y = [], []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0  # Normalize pixel values
    y = np.array(y, dtype=np.int32)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)

    # Compute class weights to handle imbalanced classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Use tf.data for efficient data loading
    AUTOTUNE = tf.data.AUTOTUNE

    # Data augmentation
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomContrast(0.1)
    ])

    def augment(image, label):
        """
        Applies the data_augmentation pipeline.
        """
        return data_augmentation(image, training=True), label

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (train_dataset
                     .shuffle(buffer_size=1000)
                     .map(augment, num_parallel_calls=AUTOTUNE)
                     .batch(batch_size)
                     .prefetch(AUTOTUNE))

    validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    validation_dataset = (validation_dataset
                          .batch(batch_size)
                          .prefetch(AUTOTUNE))

    # Define the CNN model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy"]
    )

    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=path_to_model,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )

    # Print info about the dataset
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Class distribution:", np.unique(y_train, return_counts=True))

    # Train the model
    model.fit(
        train_dataset,
        epochs=15,
        validation_data=validation_dataset,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict
    )

    # Print model summary
    model.summary()

def main():
    """
    Main function to call the train_defect_model function.
    """
    train_and_save_model_binary()

if __name__ == "__main__":
    main()
