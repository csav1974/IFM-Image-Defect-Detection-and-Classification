import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
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


def train_and_save_model_classification_new(
    DATADIR="dataCollection/Data/Perfect_Data",
    model_name="Model_20241229_classification.keras"
):
    """
    Trains a multi-class classification model for defect detection and saves the best-performing model.

    This function:
    - Loads and preprocesses image data from the specified directory.
    - Assigns class labels based on the defined defect categories.
    - Normalizes image data for training.
    - Splits the data into training and validation datasets.
    - Uses a tf.data pipeline with data augmentation to feed the model.
    - Defines and compiles a Convolutional Neural Network (CNN) model.
    - Trains the model using the training data, with validation monitoring.
    - Saves the best-performing model to the specified path.
    - Prints the model's summary and dataset statistics.
    """

    path_to_model = os.path.join("kerasModels", model_name)

    # List of categories (multi-class)
    CATEGORIES = [
        "Whiskers",
        "Chipping",
        "Scratches",
    ]

    IMG_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 20
    AUTOTUNE = tf.data.AUTOTUNE

    def list_subfolders(folder_path):
        """
        Returns a list of full paths to all subfolders in the given folder_path.
        """
        subfolders = []
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path):
                subfolders.append(full_path)
        return subfolders

    def get_image_paths_and_labels():
        """
        Gathers all image file paths and assigns labels based on the category index.
        Returns two lists: image_paths and labels.
        """
        all_image_paths = []
        all_labels = []

        folderpaths = list_subfolders(DATADIR)
        for folderpath in folderpaths:
            for category in CATEGORIES:
                category_path = os.path.join(folderpath, category)
                if not os.path.isdir(category_path):
                    # Skip if the category folder doesn't exist
                    continue

                class_index = CATEGORIES.index(category)
                
                for img_file in tqdm(
                    os.listdir(category_path),
                    desc=f"Gathering images for category: {category}"
                ):
                    img_path = os.path.join(category_path, img_file)
                    all_image_paths.append(img_path)
                    all_labels.append(class_index)

        return all_image_paths, all_labels

    def load_and_preprocess_image(path):
        """
        Loads an image file using TensorFlow operations and preprocesses it:
        - Decodes to 3 channels (RGB).
        - Converts to grayscale if desired (uncomment tf.image.rgb_to_grayscale).
        - Resizes to IMG_SIZE x IMG_SIZE.
        - Normalizes pixel values to [0,1].
        """
        image_bytes = tf.io.read_file(path)
        # Decode to 3 channels; this avoids errors with certain formats like BMP.
        image = tf.image.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image

    def parse_image(path, label):
        """
        Takes a file path and label, returns (image_tensor, label).
        """
        image = load_and_preprocess_image(path)
        return image, label

    # 1. Get paths and labels
    all_image_paths, all_labels = get_image_paths_and_labels()

    # 2. Randomly shuffle the data (keep paths and labels in sync)
    combined = list(zip(all_image_paths, all_labels))
    random.shuffle(combined)
    all_image_paths, all_labels = zip(*combined)
    all_image_paths, all_labels = list(all_image_paths), list(all_labels)

    # 3. Split into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, 
        all_labels, 
        test_size=0.3, 
        random_state=42
    )

    # Convert labels to NumPy arrays for class_weight computation
    train_labels = np.array(train_labels, dtype=np.int32)
    val_labels = np.array(val_labels, dtype=np.int32)

    # 4. Compute class weights for imbalanced classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(zip(np.unique(train_labels), class_weights))

    # 5. Create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

    # 6. Map the dataset to parse and preprocess images
    train_dataset = train_dataset.map(parse_image, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(parse_image, num_parallel_calls=AUTOTUNE)

    # 7. Data augmentation (Keras Sequential) and function to apply it
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomContrast(0.1),
    ])

    def augment(image, label):
        """
        Applies data augmentation to the image.
        """
        return data_augmentation(image, training=True), label

    # 8. Finalize the train and validation datasets
    train_dataset = (
        train_dataset
        .shuffle(buffer_size=1000)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    # 9. Define a CNN model for multi-class classification
    num_classes = len(CATEGORIES)
    model = keras.models.Sequential([
        keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        # First block
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        # Second block
        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        # Third block
        keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        # Optional fourth block
        keras.layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(1e-4)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # 10. Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["accuracy", "categorical_accuracy"]
    )

    # 11. Define callbacks
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

    # Print dataset statistics
    print("Number of training samples:", len(train_paths))
    print("Number of validation samples:", len(val_paths))
    print("Class distribution (training):", np.unique(train_labels, return_counts=True))

    # 12. Train the model
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[early_stopping, model_checkpoint],
        class_weight=class_weight_dict,
        verbose=1
    )

    # 13. Print the model summary
    model.summary()

def main():
    """
    Main function that calls train_and_save_model().
    """
    train_and_save_model_classification_new()


if __name__ == "__main__":
    main()
