import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL  import Image
from flask import Flask
from io import BytesIO

from tensorflow.python import keras
from tensorflow.python.keras import metrics, optimizers, losses
from tensorflow.python.keras.models import Sequential

import tensorflow as tf
import h5py

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def ConvModel():
    models = Sequential()
    # Convolutions
    models.add(keras.layers.Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
    models.add(keras.layers.Cropping2D(cropping=((70, 25), (0, 0)), input_shape = (160, 320, 3)))
    models.add(keras.layers.Conv2D(16, 9, strides=(4, 4), padding="same", activation='elu', name="conv1_1"))
    models.add(keras.layers.Conv2D(32, 5, strides=(2, 2), padding="same", activation='elu', name="conv1_2"))
    models.add(keras.layers.Conv2D(64, 4, strides=(1, 1), padding="same", activation='elu', name="conv1_3"))
    models.add(keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding="same"))
    
    models.add(keras.layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation='elu', name="conv2_1"))
    models.add(keras.layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation='elu', name="conv2_2"))
    models.add(keras.layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation='elu', name="conv2_3"))
    models.add(keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2, padding="same"))
    # Flatten the convolution
    models.add(keras.layers.Flatten(name="flatten"))
    # Dense layers
    models.add(keras.layers.Dense(1024, activation='elu', name="d1"))
    models.add(keras.layers.Dense(1, activation='sigmoid', name="output"))
    adam = optimizers.Adam(lr=0.000001)
    models.compile(loss="mean_squared_error", optimizer=adam, metrics=['mean_squared_error'])
    return models



class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
# 781 is good to 9
# 881
set_speed = 10
controller.set_desired(set_speed)
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        #print("throttle", throttle)
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #model = ConvModel()
    #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model = keras.models.load_model(args.model)
    print("Weight loaded.")

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


#https://machinelearningmastery.com/save-load-keras-deep-learning-models/