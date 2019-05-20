#!/usr/bin/python3
# -*- coding: utf-8 -*-

import csv

import os
import numpy as np
import random
from random import shuffle
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout


DATA_PATH = "data/driving_log.csv"
DATA_IMG = "data/"

def build_model():

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, kernel_size=(5, 5),strides=(2,2) ,activation='elu'))
    model.add(Conv2D(36, kernel_size=(5, 5),strides=(2,2) ,activation='elu'))
    model.add(Conv2D(48, kernel_size=(5, 5),strides=(2,2),activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    #model.summary()
    return model


def load_data():
    data_dir ="C:\\Users\\tompe\\Documents\\VoitureAutonome\\data"
    data_df = pd.read_csv(os.path.join(os.getcwd(),data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def main():
    """
        Main function to train the model
    """
    data = load_data()
    print("test")
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = preprocess_data.preprocess(image)
    image = np.array([image])

    steering_angle = float(model.predict(image, batch_size=1))
    throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

    # send prediction to the simulator
    send_control(steering_angle, throttle)


if __name__ == '__main__':
    main()
    print('test')

