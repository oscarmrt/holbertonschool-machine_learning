#!/usr/bin/env python3
"""Program that builds the inception network as described
in Going Deeper with Convolutions (2014)"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network as described
    in Going Deeper with Convolutions (2014)"""
    X = K.Input(shape=(224, 224, 3))
    conv = K.layers.Conv2D(64, (7, 7), padding='same',
                           activation='relu',
                           strides=2)(X)
    maxPool = K.layers.MaxPooling2D((3, 3), strides=2,
                                    padding='same')(conv)
    conv = K.layers.Conv2D(64, (1, 1), padding='same',
                           activation='relu',
                           strides=1)(maxPool)
    conv = K.layers.Conv2D(192, (3, 3), padding='same',
                           activation='relu',
                           strides=1)(conv)
    maxPool = K.layers.MaxPooling2D((3, 3), strides=2,
                                    padding='same')(conv)
    inception = inception_block(maxPool, [64, 96, 128, 16, 32, 32])
    inception = inception_block(inception, [128, 128, 192, 32, 96, 64])
    maxPool = K.layers.MaxPooling2D((3, 3), strides=2,
                                    padding='same')(inception)
    inception = inception_block(maxPool, [192, 96, 208, 16, 48, 64])
    inception = inception_block(inception, [160, 112, 224, 24, 64, 64])
    inception = inception_block(inception, [128, 128, 256, 24, 64, 64])
    inception = inception_block(inception, [112, 144, 288, 32, 64, 64])
    inception = inception_block(inception, [256, 160, 320, 32, 128, 128])
    maxPool = K.layers.MaxPooling2D((3, 3), strides=2,
                                    padding='same')(inception)
    inception = inception_block(maxPool, [256, 160, 320, 32, 128, 128])
    inception = inception_block(inception, [384, 192, 384, 48, 128, 128])
    avgPool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=1)(inception)
    drop = K.layers.Dropout(0.4)(avgPool)
    linear = K.activations.linear(drop)
    softmax = K.layers.Dense(1000, activation='softmax')(linear)
    model = K.models.Model(inputs=X, outputs=softmax)
    return model
