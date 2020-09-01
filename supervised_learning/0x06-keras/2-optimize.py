#!/usr/bin/env python3
"""Program that sets up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Function that sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics"""
    optm = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=optm,
                    metrics=['accuracy'])
    return None
