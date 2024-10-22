import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

# Define the data directory and categories
DATADIR = "dataCollection/Data/Perfect_Data/20240829"

CATEGORIES = [
    "Whiskers",
    "Chipping",
    "Scratches",
    "No_Error",
]  # This can later be changed to detect more defects

IMG_SIZE = 64  # Use the same image size as in your existing model

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # Create path to defect folders
        class_num = CATEGORIES.index(category)  # Get the classification index (0, 1, 2, 3)
        for img in tqdm(os.listdir(path)):  # Iterate over each image in the folder
            try:
                # Read the image in grayscale (assuming your existing model was trained on grayscale images)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # Resize the image to the desired size
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                training_data.append([new_array, class_num])  # Add to training data
            except Exception as e:
                pass  # Skip any corrupted images

create_training_data()

# Shuffle the data to avoid any ordering biases
random.shuffle(training_data)

X = []
y = []

# Separate features and labels
for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert to numpy arrays and normalize pixel values
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Grayscale images have 1 channel
X = X / 255.0  # Normalize to [0, 1]
y = np.array(y)

# Convert labels to one-hot encoding
y = keras.utils.to_categorical(y, num_classes=len(CATEGORIES))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation for training data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

# Data generator for validation data (no augmentation)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Create generators
train_generator = datagen.flow(X_train, y_train, batch_size=32)
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=32)

# Load your existing Keras model
existing_model_path = 'kerasModels/fullModel_v5_test.keras'  # Replace with the actual path
base_model = keras.models.load_model(existing_model_path)

# Optionally, freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new input layer
input_layer = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Manually pass the input through each layer of the base model
x = input_layer
for layer in base_model.layers:
    x = layer(x)

# Continue adding layers with unique names
x = keras.layers.Flatten(name='flatten_new')(x)
x = keras.layers.Dense(64, activation='relu', name='dense_new_1')(x)
x = keras.layers.Dropout(0.5, name='dropout_new')(x)
output_layer = keras.layers.Dense(len(CATEGORIES), activation='softmax', name='output_layer')(x)

# Create the new model
model = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Train the model using the generators
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Unfreeze the base model for fine-tuning
for layer in base_model.layers:
    layer.trainable = True

# Re-compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training with fine-tuning
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Save the model
model_name = "improvedModel_v1"
path_to_model = os.path.join("kerasModels", model_name)
model.save(f"{path_to_model}.keras")

# Display the model's architecture
model.summary()
