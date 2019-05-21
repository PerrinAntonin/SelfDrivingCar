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
        self.alea = Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3))
        self.conv1_1 = keras.layers.Conv2D(32, 4, activation='relu', name="conv1_1")
        self.conv1_2 = keras.layers.Conv2D(64, 3, activation='relu', name="conv1_2")
        self.pool1 = keras.layers.MaxPooling2D((2, 2))

        self.conv2_1 = keras.layers.Conv2D(128, 3, activation='relu', name="conv2_1")
        self.conv2_2 = keras.layers.Conv2D(128, 3, activation='relu', name="conv2_2")
        self.pool2 = keras.layers.MaxPooling2D((2, 2))

        self.conv3_1 = keras.layers.Conv2D(128, 3, activation='relu', name="conv3_1")
        self.conv3_2 = keras.layers.Conv2D(128, 3, activation='relu', name="conv3_2")
        self.conv3_3 = keras.layers.Conv2D(128, 3, activation='relu', name="conv3_3")
        self.pool3 = keras.layers.MaxPooling2D((2, 2))

        self.conv4_1 = keras.layers.Conv2D(128, 3, activation='relu', name="conv4_1")
        self.conv4_2 = keras.layers.Conv2D(128, 3, activation='relu', name="conv4_2")
        self.conv4_3 = keras.layers.Conv2D(128, 3, activation='relu', name="conv4__3")
        self.pool4 = keras.layers.MaxPooling2D((2, 2))

        self.conv5_1 = keras.layers.Conv2D(128, 3, activation='relu', name="conv5_1")
        self.conv5_2 = keras.layers.Conv2D(128, 3, activation='relu', name="conv5_2")
        self.conv5_3 = keras.layers.Conv2D(128, 3, activation='relu', name="conv5_3")
        self.pool5 = keras.layers.MaxPooling2D((2, 2))

        # Flatten the convolution
        self.flatten = keras.layers.Flatten(name="flatten")
        # Add layers
        self.d1 = keras.layers.Dense(256, activation='relu', name="d1")
        self.d2 = keras.layers.Dense(128, activation='relu', name="d2")
        self.out = keras.layers.Dense(10, activation='softmax', name="output")

    def call(self, image):
        alea = self.alea(image)
        conv1_1 = self.conv1_1(alea)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 =self.pool1(conv1_2)
                
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv1_1)
        pool2 =self.pool2(conv2_2)
        
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 =self.pool3(conv3_3)
        
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 =self.pool4(conv4_3)
        
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        pool5 =self.pool1(conv5_3)

        flatten = self.flatten(pool5)
        d1 = self.d1(flatten)
        d2 = self.d2(d1)
        output = self.out(d2)
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


