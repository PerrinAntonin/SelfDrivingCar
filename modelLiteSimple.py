import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import cv2
import os
import numpy as np
import random
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python import keras
from tensorflow.python.keras import metrics, optimizers, losses
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout,MaxPooling2D
import tensorflow as tf



assert hasattr(tf, "function") # Be sure to use tensorflow 2.0
DATA_PATH = "data/"
DATA_IMG = "data/"
def load_data():
    print("search data...")
    data_df = pd.read_csv(os.path.join(os.getcwd(),DATA_PATH, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
    print("data find")
    return X_train, X_test, y_train, y_test

def get_data(batch_size,imageData, rotationData):
    print("load data...")
    rotations = []
    images = []
    angle_correction = [0., 0.25, -0.25] # [Center, Left, Right]
    print("size",imageData.shape)
    for index in range(0,batch_size):
        
        i = random.choice([0, 1, 2]) # for direction
        img = cv2.imread(os.path.join(DATA_IMG,imageData[index][i]).replace(" ", ""))
        #converti la couleur de l'image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #recupere la rotation 
        rotation = float(rotationData[index])
        #ajou de la correction
        rotation = rotation + angle_correction[i]
        #ajou des informations a la liste
        images.append(img)
        rotations.append(rotation)

    print("il y a eu",len(images),"images de chargées")
    return np.array(images), np.array(rotations)

# création du reseaux convolutif
def ConvModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    model.add(Conv2D(8, 9, strides=(4, 4), padding="same", activation="elu"))
    model.add(Conv2D(16, 5, strides=(2, 2), padding="same", activation="elu"))
    model.add(Conv2D(32, 4, strides=(1, 1), padding="same", activation="elu"))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2, padding="same"))
    
    model.add(Conv2D(32, 3, strides=(1, 1), padding="same", activation='elu'))
    model.add(Conv2D(32, 3, strides=(1, 1), padding="same", activation='elu'))
    model.add(Conv2D(32, 3, strides=(1, 1), padding="same", activation='elu'))
    model.add(MaxPooling2D(pool_size=[2, 2], strides=2, padding="same"))
    model.add(Flatten())
    model.add(Dropout(.6))
    model.add(Dense(1024, activation="elu"))
    model.add(Dropout(.3))
    model.add(Dense(1))

    #adamperso = optimizers.Adam(lr=0.000001)
    model.compile(loss="mean_squared_error", optimizer="adam")
    
    return model



def main():
    data = load_data()

    x_train = data[0]
    x_valid = data[1]
    y_train = data[2]
    y_valid = data[3]

    # Peut etre a retirer next
    images, rotations = get_data(2240,x_train,y_train)
    images_valid, rotations_valid = get_data(384,x_valid,y_valid)

    # conversion des images de float 64 en 32 car con2d veut du 32
    images = images.astype(np.float32)
    # conversion des images de float 64 en 32 car con2d veut du 32
    images_valid = images_valid.astype(np.float32)
    #affichage d'un exemple
    plt.title(rotations[57])
    plt.imshow(images[57]/255)
    plt.show()

    print("rotation shape",rotations.shape)
    print("image shape",images.shape)
    print("image validation shape",images_valid.shape)

    # Normalisation
    print("Moyenne et ecart type des images", images.mean(), images.std())
    #scaler = StandardScaler()
    #scaled_images = scaler.fit_transform(images.reshape(-1, 160*320))
    #scaled_images_valid = scaler.transform(images_valid.reshape(-1, 160*320))
    #print("Moyenne et ecart type des images normalisé", scaled_images.mean(), scaled_images.std())

    scaled_images = images.reshape(-1, 160,320,3)
    scaled_images_valid = images_valid.reshape(-1, 160,320,3)
    print("scaled images after normalisation",scaled_images.shape)

    #train_dataset = tf.data.Dataset.from_tensor_slices((scaled_images, rotations))
    #train_dataset = train_dataset.shuffle(buffer_size=1280)
    #train_dataset = train_dataset.shuffle(1280 * 50)
    #valid_dataset = tf.data.Dataset.from_tensor_slices((scaled_images_valid, rotations_valid))


    model = ConvModel()
    model.summary()
    BATCH_SIZE = 64
    model.fit(scaled_images, rotations,
     batch_size=BATCH_SIZE,
      epochs=12,
     validation_data=(scaled_images_valid, rotations_valid))

    model.save('model.h5')
    print("Saved model to disk")

if __name__ == '__main__':
    main()