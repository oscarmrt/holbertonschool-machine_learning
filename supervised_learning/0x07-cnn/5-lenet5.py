#!/usr/bin/env python3
"""Program that builds a modified version of
the LeNet-5 architecture using keras"""
import tensorflow.keras as K


def lenet5(X):
    """Function that builds a modified version of
    the LeNet-5 architecture using keras"""
    conv1 = K.layers.Conv2D(6, kernel_size=(5, 5),
                            activation='relu', kernel_initializer='he_normal',
                            padding="same")(X)
    # Pool layer 1
    maxpool1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    # C2 convolutional layer
    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5),
                            activation='relu', kernel_initializer='he_normal',
                            padding='Valid')(maxpool1)
    # Pool layer 2
    maxpool2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    # Flatten the CNN output so that we can connect it with FC layers
    flatten = K.layers.Flatten()(maxpool2)
    # FC Fully Connected Layer 120 nodes
    dense1 = K.layers.Dense(120, activation='relu',
                            kernel_initializer='he_normal')(flatten)
    # FC Fully Connected Layer 84 nodes
    dense2 = K.layers.Dense(84, activation='relu',
                            kernel_initializer='he_normal')(dense1)
    # Output Layer with softmax activation
    dense3 = K.layers.Dense(10, activation='softmax',
                            kernel_initializer='he_normal')(dense2)
    model = K.Model(inputs=X, outputs=dense3)
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(), metrics=['accuracy'])
    return model
