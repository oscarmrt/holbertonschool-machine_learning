#!/usr/bin/env python3
"""Program that builds the DenseNet-121 architecture as
described in Densely Connected Convolutional Networks"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Function that builds the DenseNet-121 architecture as
    described in Densely Connected Convolutional Networks"""
    XInput = K.Input(shape=(224, 224, 3))
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]
    x = K.layers.BatchNormalization(axis=3)(XInput)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(nb_filter, kernel_size=(7, 7), strides=(2, 2),
                        kernel_initializer='he_normal', padding='same')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    for idx in range(len(nb_layers) - 1):
        x, nb_filter = dense_block(x, nb_filter, growth_rate, nb_layers[idx])
        x, nb_filter = transition_layer(x, nb_filter, compression)
    x, nb_filter = dense_block(x, nb_filter, growth_rate, nb_layers[idx + 1])
    x = K.layers.AveragePooling2D((7, 7))(x)
    x = K.layers.Dense(1000, activation='softmax')(x)
    model = K.Model(inputs=XInput, outputs=x)
    return model
