#!/usr/bin/env python3
"""Program that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    n = len(layers)
    for x in range(n):
        if x == 0:
            model.add(K.layers.Dense(layers[x], activation=activations[x],
                      kernel_regularizer=L2, input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layers[x], activation=activations[x],
                      kernel_regularizer=L2))
        if x < n - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
