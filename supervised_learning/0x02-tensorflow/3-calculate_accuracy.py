#!/usr/bin/env python3
"""Program that calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Function that calculates the accuracy of a prediction"""
    equal = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equal, float))
    return accuracy
