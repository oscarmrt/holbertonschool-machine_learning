#!/usr/bin/env python3
"""Program that creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Function that creates a layer of a neural network using dropout"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.layers.Dropout(keep_prob)
    mod = tf.layers.Dense(n, activation, kernel_initializer=init,
                          kernel_regularizer=reg, name='layer')
    return mod(prev)
