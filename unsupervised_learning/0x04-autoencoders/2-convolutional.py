#!/usr/bin/env python3
"""Program creates a convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder"""
    inputs = keras.Input(shape=(input_dims))
    encoded = inputs
    for f in filters:
        encoded = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                      padding='same')(encoded)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = keras.Model(inputs, encoded)
    latentInputs = keras.Input(shape=(latent_dims))
    x = latentInputs
    fr = filters.copy()
    fr.reverse()
    for f in range(len(fr)):
        if (f == len(fr) - 1):
            x = keras.layers.Conv2D(filters[f], (3, 3), activation='relu',
                                    padding='valid')(x)
        else:
            x = keras.layers.Conv2D(filters[f], (3, 3), activation='relu',
                                    padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(input_dims[-1], (3, 3), activation='sigmoid',
                                  padding='same')(x)
    decoder = keras.Model(latentInputs, decoded)
    auto = keras.Model(inputs, decoder(encoder(inputs)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
