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

# Pfad zum Datensatz
DATADIR = "dataCollection/Data/training20241218_firstStep"
model_name = "Model_20241220_firstStep.keras"
path_to_model = os.path.join("kerasModels", model_name)


CATEGORIES = [
    "Defect_Default",
    "No_Error",
] 

IMG_SIZE = 128
batch_size = 16

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                # Bild laden in Farbe (BGR)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # Resize
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    return training_data

training_data = create_training_data()

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
X = X / 255.0
y = np.array(y, dtype=np.int32)

# Trainings- und Validierungsdaten Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)

# Klassen-Gewichte berechnen, falls unbalanciert
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Daten-Pipeline mit tf.data
AUTOTUNE = tf.data.AUTOTUNE

# Datenaugmentation per Preprocessing-Layer
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomContrast(0.1)
])

def augment(image, label):
    return data_augmentation(image, training=True), label

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1000).map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
validation_dataset = validation_dataset.batch(batch_size).prefetch(AUTOTUNE)

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
    keras.layers.Dense(len(CATEGORIES), activation='softmax')
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=path_to_model,
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Klassenverteilung:", np.unique(y_train, return_counts=True))

model.fit(
    train_dataset,
    epochs=15,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weight_dict
)

model.save(f"{path_to_model}.keras")
model.summary()