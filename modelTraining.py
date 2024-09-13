import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm
import keras


# the path is fixed because it is only used for training the data and will not be present in the finisihed programm
DATADIR = "dataCollection/trainingdata"

CATEGORIES = ["Whiskers", "Chipping", "No_Error"] # This can later be changed to detect more defects

IMG_SIZE = 32

training_data = []


def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path to defect folders
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1, 2). 0=Wiskers 1=Chipping 2=NoDefekt

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

create_training_data()


random.shuffle(training_data) # to avoid unwanted learning behaviour when only lerning one defect at a time

X = []
y = []

for features,label in training_data:
    X.append(features)
    if label == 0:
        y.append([1, 0, 0])  # [1, 0, 0] für Kategorie 1
    elif label == 1:
        y.append([0, 1, 0])  # [0, 1, 0] für Kategorie 2
    elif label == 2:
        y.append([0, 0, 1])  # [0, 0, 1] für Kategorie 3

#print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1)) # just for checking

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0
y = np.array(y)

# the following part creates the actual model with its multiple layers


# Erstellen eines Sequential-Modells
model = keras.models.Sequential()

# Erste Convolutional-Schicht
model.add(keras.layers.Conv2D(256, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Zweite Convolutional-Schicht
model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten-Schicht zur Umwandlung von 3D-Feature-Maps in 1D-Feature-Vektoren
model.add(keras.layers.Flatten())

# Dense-Schicht
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))

# Drei Ausgabeschichten, eine für jede Kategorie
model.add(keras.layers.Dense(3))
model.add(keras.layers.Activation('softmax'))

# Kompilierung des Modells mit der loss='categorical_crossentropy'
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training des Modells
model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3)


# Display the model's architecture

model_name = "firstModelTest" 
path_to_model = os.path.join("kerasModels", model_name)
model.save(f"{path_to_model}.keras")

model.summary()

"""

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(256, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(keras.layers.Dense(64))

model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3) # actual training


"""