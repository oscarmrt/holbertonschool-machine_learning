#!/usr/bin/env python3
"""Program that creates a tensorflow layer that includes
L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a tensorflow layer that includes
    L2 regularization"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    mod = tf.layers.Dense(n, activation, kernel_initializer=init,
                          kernel_regularizer=reg, name='layer')
    return mod(prev)
