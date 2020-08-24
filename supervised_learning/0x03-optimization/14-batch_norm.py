#!/usr/bin/env python3
"""Program creates a batch normalization layer
for a neural network in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Program creates a batch normalization layer
    for a neural network in tensorflow"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    y = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')
    x = y(prev)
    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8
    norma = tf.nn.batch_normalization(x, mean, variance, beta, gamma, epsilon)
    return activation(norma)
