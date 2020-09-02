#!/usr/bin/env python3
"""Program that saves a model’s configuration in JSON format
and loads a model with a specific configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration in JSON format"""
    nw = network.to_json()
    with open(filename, "w") as jFile:
        jFile.write(nw)
    return None


def load_config(filename):
    """Function that  loads a model with a specific configuration"""
    with open(filename, "r") as jFile:
        jsonLoaded = jFile.read()
    loadedModel = K.models.model_from_json(jsonLoaded)
    return loadedModel
