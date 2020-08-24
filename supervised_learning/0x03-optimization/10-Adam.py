#!/usr/bin/env python3
"""Program that creates the training operation for a neural network
in tensorflow using the Adam optimization algorithm"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Function that creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    opt = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
    return opt
