import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# The path is fixed because it is only used for training the data and will not be present in the final program
DATADIR = "dataCollection/Data/Training20240829_v3"

CATEGORIES = [
    "Whiskers",
    "Chipping",
    "Scratches",
    "No_Error",
]  # This can later be changed to detect more defects

IMG_SIZE = 128

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Create path to defect folders
        class_num = CATEGORIES.index(category)  # Get the classification index (0, 1, 2, 3)
        for img in tqdm(os.listdir(path)):  # Iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)  # Resize to normalize data size
                training_data.append([new_array, class_num])  # Add this to our training_data
            except Exception as e:
                pass  # Ignore errors and continue
    return training_data

training_data = create_training_data()

random.shuffle(training_data)  # To avoid unwanted learning behavior when learning only one defect type at a time

X = []
y = []

for features, label in training_data:
    X.append(features)
    one_hot_label = [0] * len(CATEGORIES)  # Initialize one-hot label
    one_hot_label[label] = 1  # Set the correct category to 1
    y.append(one_hot_label)  # Append the one-hot label

# Convert and normalize data
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Reshape for the model input
X = X / 255.0  # Normalize pixel values
y = np.array(y)  # Convert labels to numpy array

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Data augmentation for training data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

# Data generator for validation data (no augmentation)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

batch_size = 32  # Define batch size

# Create generators
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

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

# Compile the model with loss='categorical_crossentropy'
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Train the model using the generators
model.fit(
    train_generator,
    epochs=12,
    validation_data=validation_generator,
    callbacks=[early_stopping],
)

# Save the model
model_name = "Model_20240829_v4"
path_to_model = os.path.join("kerasModels", model_name)
model.save(f"{path_to_model}.keras")

# Display the model's architecture
model.summary()
