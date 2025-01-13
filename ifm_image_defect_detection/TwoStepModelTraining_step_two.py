import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def train_and_save_model_classification(DATADIR = "dataCollection/Data/Perfect_Data", model_name = "Model_20241229_classification.keras"):
    """
    Trains a multi-class classification model for defect detection and saves the best-performing model.

    This function:
    - Loads and preprocesses image data from the specified directory.
    - Assigns class labels based on the defined defect categories.
    - Normalizes and reshapes the image data for training.
    - Splits the data into training and validation datasets.
    - Applies data augmentation to the training data.
    - Defines and compiles a Convolutional Neural Network (CNN) model for multi-class classification.
    - Trains the model using the training data, with validation monitoring.
    - Saves the best-performing model to the specified file path.
    - Prints the model's summary and dataset statistics.

    Parameters:
    - DATADIR (str): Path to the directory containing the categorized image data.
    - model_name (str): Name of the file to save the trained model.

    Returns:
    None
    """

    path_to_model = os.path.join("kerasModels", model_name)

    # Categories in the dataset
    CATEGORIES = [
        "Whiskers",
        "Chipping",
        "Scratches",
    ] 

    # Image settings
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
        Creates and returns a list of (image_array, class_label) for all images
        in the dataset folders.
        """
        folderpaths = list_subfolders(DATADIR)
        training_data = []
        
        for folderpath in folderpaths:
            for category in CATEGORIES:
                path = os.path.join(folderpath, category)
                
                # Skip if folder does not exist
                if not os.path.isdir(path):
                    continue
                
                class_num = CATEGORIES.index(category)
                for img in tqdm(os.listdir(path), desc=f"Loading {category}"):
                    try:
                        img_array = cv2.imread(
                            os.path.join(path, img), 
                            cv2.IMREAD_GRAYSCALE
                        )
                        
                        # Resize image
                        new_array = cv2.resize(
                            img_array, 
                            (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_LINEAR
                        )
                        
                        training_data.append([new_array, class_num])
                    except Exception:
                        pass
        
        return training_data

    # Create training data
    training_data = create_training_data()
    random.shuffle(training_data)

    # Separate features and labels
    X = []
    y = []
    for features, label in training_data:
        X.append(features)
        y.append(label)

    # Convert to NumPy arrays and reshape
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X / 255.0  # Normalize
    y = np.array(y, dtype=np.int32)

    # Split into training and validation data
    X_train, X_val, y_train, y_val = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=42
    )

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)

    # Compute class weights for imbalanced datasets
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # Create a data pipeline using tf.data
    AUTOTUNE = tf.data.AUTOTUNE

    # Data augmentation using Keras layers
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomContrast(0.1)
    ])

    def augment(image, label):
        """
        Applies data augmentation to the given image.
        """
        return data_augmentation(image, training=True), label

    # Create TensorFlow datasets
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

    # Build the model
    model = keras.models.Sequential([
        keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        keras.layers.Conv2D(
            32, (3, 3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(
            64, (3, 3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(
            128, (3, 3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(
            256, (3, 3), 
            activation='relu', 
            padding='same', 
            kernel_initializer='he_normal'
        ),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(CATEGORIES), activation='softmax')
    ])

    # Compile the model
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

    # Print training set shape and class distribution
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Class distribution (train):", np.unique(y_train, return_counts=True))

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
    Main function that calls train_and_save_model().
    """
    train_and_save_model_classification()


if __name__ == "__main__":
    main()
