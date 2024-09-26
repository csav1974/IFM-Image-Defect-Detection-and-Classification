import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
import tensorflow as tf
import keras

# The path is fixed because it is only used for training the data and will not be present in the final program
DATADIR = "dataCollection/Data/trainingData_v5"

CATEGORIES = [
    "Whiskers",
    "Chipping",
    "Scratches",
    "No_Error",
]  # This can later be changed to detect more defects

IMG_SIZE = 32

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Create path to defect folders
        class_num = CATEGORIES.index(category)  # Get the classification index (0, 1, 2, 3)
        for img in tqdm(os.listdir(path)):  # Iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)  # Resize to normalize data size
                training_data.append([new_array, class_num])  # Add this to our training_data
            except Exception as e:  # In the interest of keeping the output clean...
                pass

create_training_data()

random.shuffle(training_data)  # To avoid unwanted learning behavior when learning only one defect type at a time

X = []
y = []

for features, label in training_data:
    X.append(features)
    if label == 0:
        y.append([1, 0, 0, 0])  # [1, 0, 0, 0] for category 1
    elif label == 1:
        y.append([0, 1, 0, 0])  # [0, 1, 0, 0] for category 2
    elif label == 2:
        y.append([0, 0, 1, 0])  # [0, 0, 1, 0] for category 3
    elif label == 3:
        y.append([0, 0, 0, 1])  # [0, 0, 0, 1] for category 4

# Convert and normalize data
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0
y = np.array(y)

# Create a Sequential model
model = keras.models.Sequential()

# First convolutional layer
model.add(keras.layers.Conv2D(256, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
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

# Four output layers, one for each category
model.add(keras.layers.Dense(4))
model.add(keras.layers.Activation("softmax"))

# Compile the model with loss='categorical_crossentropy'
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

# Train the model
model.fit(X, y, batch_size=32, epochs=15, validation_split=0.3)

# Save the model
model_name = "fullModel_v2"
path_to_model = os.path.join("kerasModels", model_name)
model.save(f"{path_to_model}.keras")

# Display the model's architecture
model.summary()
