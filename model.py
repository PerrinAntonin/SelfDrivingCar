#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Just disables the warning, doesn't enable AVX/FMA
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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Lambda, Dropout
from tensorflow.python import keras
from tensorflow.python.keras import metrics, optimizers, losses
import tensorflow as tf


assert hasattr(tf, "function") # Be sure to use tensorflow 2.0
DATA_PATH = "data/"

def load_data():
    data_df = pd.read_csv(os.path.join(os.getcwd(),DATA_PATH, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train[10][2])
    plt.imshow(X_train[10][2])
    return X_train, X_test, y_train, y_test

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
        self.out = keras.layers.Dense(1, activation='softmax', name="output")
        

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




data = load_data()
"""
model = ConvModel()

x_train = data[0]
y_train = data[2]
x_valid = data[1]
y_valid = data[3]

x_train = x_train.astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))


BATCH_SIZE = 64

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
#track the evolution
# Loss
train_loss = metrics.Mean(name='train_loss')
valid_loss = metrics.Mean(name='valid_loss')
# Accuracy
train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_accuracy = metrics.SparseCategoricalAccuracy(name='valid_accuracy')


@tf.function
def train_step(image, targets):
    # permet de surveiller les opérations réalisé afin de calculer le gradient
    with tf.GradientTape() as tape:
        # fait une prediction
        predictions = model(image)
        # calcul de l'erreur e nfonction de la prediction et des targets
        loss = loss_object(targets, predictions)
    # calcul du gradient en fonction du loss
    # trainable_variables est la lst des variable entrainable dans le model
    gradients = tape.gradient(loss, model.trainable_variables)
    # changement des poids grace aux gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # ajout de notre loss a notre vecteur de stockage
    train_loss(loss)
    train_accuracy(targets, predictions)

@tf.function
# vérifier notre accuracy de notre model afin d'evité l'overfitting
def valid_step(image, targets):
    predictions = model(image)
    t_loss = loss_object(targets, predictions)
    # mets a jour les metrics
    valid_loss(t_loss)
    valid_accuracy(targets, predictions)
        

epoch = 10
batch_size = 32
b = 0
for epoch in range(epoch):
    # Training set
    for images_batch, targets_batch in train_dataset.batch(batch_size):
        train_step(images_batch, targets_batch)
        template = '\r Batch {}/{}, Loss: {}, Accuracy: {}'
        print(template.format(
            b, len(targets), train_loss.result(), 
            train_accuracy.result()*100
        ), end="")
        b += batch_size
    # Validation set
    for images_batch, targets_batch in valid_dataset.batch(batch_size):
        valid_step(images_batch, targets_batch)

    template = '\nEpoch {}, Valid Loss: {}, Valid Accuracy: {}'
    print(template.format(
        epoch+1,
        valid_loss.result(), 
        valid_accuracy.result()*100)
    )
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    train_accuracy.reset_states()
    train_loss.reset_states()
"""