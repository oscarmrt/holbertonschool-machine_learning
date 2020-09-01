#!/usr/bin/env python3
"""Program that builds a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    n = len(layers)
    for x in range(n):
        if x == 0:
            output = K.layers.Dense(layers[x],
                                    activation=activations[x],
                                    kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[x],
                                    activation=activations[x],
                                    kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=output)
