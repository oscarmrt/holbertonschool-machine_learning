#!/usr/bin/env python3
"""Program that trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """Function that trains a model using mini-batch gradient descent"""
    callback = []
    if early_stopping and validation_data:
        callback = [K.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=patience)]
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              verbose=verbose, epochs=epochs, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=callback)
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """Function keeps the initial learning rate for the first
            ten epochs and decreases it exponentially after that"""
            return alpha / (1 + decay_rate * epoch)
        callback.append(K.callbacks.LearningRateScheduler(scheduler,
                                                          verbose=1))
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              verbose=verbose, epochs=epochs, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=callback)
    else:
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, validation_data=validation_data,
                              verbose=verbose, shuffle=shuffle)
    return history
