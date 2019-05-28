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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Lambda, Dropout
from tensorflow.python import keras
from tensorflow.python.keras import metrics, optimizers, losses
import tensorflow as tf


from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Cropping2D, Lambda, Dropout


assert hasattr(tf, "function") # Be sure to use tensorflow 2.0
DATA_PATH = "data/"
DATA_IMG = "data/"
def load_data():
    print("search data...")
    data_df = pd.read_csv(os.path.join(os.getcwd(),DATA_PATH, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

def get_data(batch_size,imageData, rotationData):
    print("load data...")
    rotations = []
    images = []
    
    angle_correction = [0., 0.25, -.25] # [Center, Left, Right]

    for index in range(0,batch_size):
        i = random.choice([0, 1, 2]) # for direction

        img = cv2.imread(os.path.join(DATA_IMG,imageData[index][i]).replace(" ", ""))
        if img is None: 
            continue

        #converti la couleur de l'image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = img.astype(np.float32)
        #recupere la rotation 
        if index%3==0:
            subIndex=index//3
            rotation = float(rotationData[subIndex])

        #ajou de la correction
        rotation = rotation + angle_correction[i]

        #ajou des informations a la liste
        images.append(img)
        rotations.append(rotation)

    print("end loading longeur image:",len(images))

    return np.array(images), np.array(rotations)

# création du reseaux convolutif
# création du reseaux convolutif
class ConvModel(keras.Model):

    def __init__(self):
        super(ConvModel, self).__init__()
        # Convolutions
        self.conv1 = keras.layers.Conv2D(32, 4, activation='relu', name="conv1")
        self.conv2 = keras.layers.Conv2D(64, 3, activation='relu', name="conv2")
        self.conv3 = keras.layers.Conv2D(128, 3, activation='relu', name="conv3")
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




data = load_data()

model = ConvModel()

x_train = data[0]
x_valid = data[1]
y_train = data[2]
y_valid = data[3]

# Peut etre a retirer next
images, rotations = get_data(32,x_train,y_train)
images_valid, rotations_valid = get_data(32,x_valid,y_valid)

# conversion des images de float 64 en 32 car con2d veut du 32
#images = images.astype(np.float32)
# conversion des images de float 64 en 32 car con2d veut du 32
#images_valid = images_valid.astype(np.float32)
    
plt.title(rotations[10])
plt.imshow(images[10], cmap="gray")
plt.show()

images=np.array(images)
rotations = np.array(rotations)
print("rotation shape",rotations.shape)
images_valid=np.array(images_valid)
print("image before shape",images.shape)
images = np.reshape(images,(-1, 160,320,3))
images_valid = np.reshape(images_valid,(-1, 160,320,3))
print("image after shape",images.shape)

# Normalisation
print("Moyenne et ecart type des images", images.mean(), images.std())
scaler = StandardScaler()
scaled_images = scaler.fit_transform(images.reshape(-1, 160*320))
scaled_images_valid = scaler.transform(images_valid.reshape(-1, 160*320))
print("Moyenne et ecart type des images normalisé", scaled_images.mean(), scaled_images.std())
print("after normalisation",rotations.shape)
print("after normalisation",scaled_images.shape)

scaled_images = scaled_images.reshape(-1, 160,320,3)
scaled_images_valid = scaled_images_valid.reshape(-1, 160,320,3)
print("after scaled",rotations.shape)
print("after scaled",scaled_images.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((scaled_images, rotations))
valid_dataset = tf.data.Dataset.from_tensor_slices((scaled_images_valid, rotations_valid))


BATCH_SIZE = 32

loss_object = keras.losses.SparseCategoricalCrossentropy()
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

        print("after creation model targets shape",targets)
        print("after creation model prediction shape",predictions)
        # calcul de l'erreur en fonction de la prediction et des targets
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


model.save("model.h5")

