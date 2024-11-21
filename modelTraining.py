import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# The path is fixed because it is only used for training the data and will not be present in the final program
DATADIR = "dataCollection/Data/detectedErrors/machinefoundErrors/20241024_A3-1"

CATEGORIES = [
    "Whiskers",
    "Chipping",
    "Scratches",
    "No_Error",
]  # This can later be changed to detect more defects

IMG_SIZE = 128
batch_size = 32

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Create path to defect folders
        class_num = CATEGORIES.index(category)  # Get class number (0, 1, 2, 3)
        for img in tqdm(os.listdir(path)):  # Iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert to array
                new_array = cv2.resize(
                    img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
                )  # Resize to normalize data size
                training_data.append([new_array, class_num])  # Add to training data
            except Exception as e:
                pass  # Ignore errors and continue
    return training_data

training_data = create_training_data()

random.shuffle(training_data)  # To avoid unwanted learning behavior when learning only one defect type at a time

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)  # Keep integer labels

# Convert and normalize data
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for model input
X = X / 255.0  # Normalize pixel values
y = np.array(y, dtype=np.int32)  # Convert labels to NumPy array with integer type

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Ensure labels are of integer type
y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Data augmentation for training data using tf.data and augmentation layers
AUTOTUNE = tf.data.AUTOTUNE

def augment(image, label):
    # Data augmentation transformations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.batch(batch_size).prefetch(AUTOTUNE)

# Create a Sequential model
model = keras.models.Sequential()

# First convolutional layer
model.add(keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Second convolutional layer
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to convert 3D feature maps into 1D feature vectors
model.add(keras.layers.Flatten())

# Dense layer
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation("relu"))

# Output layer with the number of categories
model.add(keras.layers.Dense(len(CATEGORIES)))
model.add(keras.layers.Activation("softmax"))

# Compile the model with loss='sparse_categorical_crossentropy'
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Train the model using the datasets
model.fit(
    train_dataset,
    epochs=12,
    validation_data=validation_dataset,
    callbacks=[early_stopping],
    class_weight=class_weight_dict,
)

# Save the model
model_name = "Model_v8"
path_to_model = os.path.join("kerasModels", model_name)
model.save(f"{path_to_model}.keras")

# Display the model's architecture
model.summary()
