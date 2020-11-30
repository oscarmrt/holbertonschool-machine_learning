#!/usr/bin/env python3
"""Program that creates a sparse autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Function that creates a sparse autoencoder"""
    inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    regularizer = keras.regularizers.l1(lambtha)
    encoded = keras.layers.Dense(latent_dims, activation='relu',
                                 activity_regularizer=regularizer)(encoded)
    encoder = keras.Model(inputs, encoded)
    latentInputs = keras.Input(shape=(latent_dims,))
    decoded = latentInputs
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.Model(latentInputs, decoded)
    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
