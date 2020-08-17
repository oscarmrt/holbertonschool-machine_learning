#!/usr/bin/env python3
"""Program that returns the tensor output of the layer"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Function that returns the tensor output of the layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y = tf.layers.Dense(n, activation, kernel_initializer=init, name='layer')
    return y(prev)
