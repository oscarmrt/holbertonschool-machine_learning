#!/usr/bin/env python3
"""Program that builds an identity block as described in Deep
Residual Learning for Image Recognition (2015)"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Function that builds an identity block as described in Deep
    Residual Learning for Image Recognition (2015)"""
    F11 = filters[0]
    F3 = filters[1]
    F12 = filters[2]
    cpA = A_prev
    A_prev = K.layers.Conv2D(F11, kernel_size=(1, 1), strides=(1, 1),
                             padding='valid',
                             kernel_initializer='he_normal')(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Activation('relu')(A_prev)
    A_prev = K.layers.Conv2D(F3, kernel_size=(3, 3), strides=(1, 1),
                             padding='same',
                             kernel_initializer='he_normal')(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Activation('relu')(A_prev)
    A_prev = K.layers.Conv2D(F12, kernel_size=(1, 1), strides=(1, 1),
                             padding='valid',
                             kernel_initializer='he_normal')(A_prev)
    A_prev = K.layers.BatchNormalization(axis=3)(A_prev)
    A_prev = K.layers.Add()([A_prev, cpA])
    A_prev = K.layers.Activation('relu')(A_prev)
    return A_prev
