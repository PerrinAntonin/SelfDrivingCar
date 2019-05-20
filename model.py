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
from tensorflow.python import keras


DATA_PATH = "data/"
DATA_IMG = "data/"

# création du reseaux convolutif
class ConvModel(keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        # Convolutions
        self.alea = Lambda(lambda x: x/127.5-1.0, input_shape=(320*160 Inpu))
        self.conv1 = keras.layers.Conv2D(32, 4, activation='relu', name="conv1")
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu', name="conv2")
        self.conv3 = keras.layers.Conv2D(128, 3, activation='relu', name="conv3")

        self.pool1 = keras.layers.MaxPooling2D((2, 2))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.summary()

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        model.summary()
        # Flatten the convolution
        self.flatten = keras.layers.Flatten(name="flatten")
        # Dense layers
        self.d1 = keras.layers.Dense(128, activation='relu', name="d1")
        self.out = keras.layers.Dense(10, activation='softmax', name="output")

    def call(self, image):
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        flatten = self.flatten(conv3)
        d1 = self.d1(flatten)
        output = self.out(d1)
        return output


def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(),DATA_PATH, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test




def main():
    """
        Main function to train the model
    """
    data = load_data()
    print(data[3])



if __name__ == '__main__':
    main()


