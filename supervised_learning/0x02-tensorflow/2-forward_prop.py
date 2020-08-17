#!/usr/bin/env python3
"""Program that creates the forward propagation graph
for the neural network"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Function that creates the forward propagation graph
    for the neural network"""
    firstLayer = create_layer(x, layer_sizes[0], activations[0])
    layer = firstLayer
    for x in range(1, len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[x], activations[x])
    return layer
