#!/usr/bin/env python3
"""Script that trains a convolutional neural network
to classify the CIFAR 10 dataset"""
import os
import tensorflow.keras as K
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def preprocess_data(X, Y):
    """Function that trains a convolutional neural network
    to classify the CIFAR 10 dataset"""
    X = X / 255.0
    Y = K.utils.to_categorical(Y, 10)
    return X, Y


(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
base_model = K.applications.ResNet50(weights='imagenet', include_top=False,
                                     input_shape=(200, 200, 3))
base_model.trainable = False
model = K.Sequential()
model.add(K.layers.UpSampling2D((2, 2)))
model.add(K.layers.UpSampling2D((2, 2)))
model.add(K.layers.UpSampling2D((2, 2)))
model.add(base_model)
model.add(K.layers.Flatten())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(64, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(10, activation='softmax'))
model.compile(optimizer=K.optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=20,
                    validation_data=(x_test, y_test))
model.save('cifar10.h5')
