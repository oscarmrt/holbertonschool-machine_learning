#!/usr/bin/env python3
"""Program that builds a transition layer as described in
Densely Connected Convolutional Networks"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Function that builds a transition layer as described in
    Densely Connected Convolutional Networks"""
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(int(nb_filters * compression), kernel_size=(1, 1),
                        kernel_initializer='he_normal', padding='same')(X)
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)
    return X, int(nb_filters * compression)
