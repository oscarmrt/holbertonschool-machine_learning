#!/usr/bin/env python3
"""Program that loads and saves a model’s weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """Function that saves a model’s weights"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Function that loads a model’s weights"""
    network.load_weights(filename)
    return None
