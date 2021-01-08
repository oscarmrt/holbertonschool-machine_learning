#!/usr/bin/env python3
"""Class SelfAttention that inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Class SelfAttention"""
    def __init__(self, units):
        """Class constructor"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Public instance method that returns context, weights"""
        s_prev = tf.expand_dims(s_prev, 1)
        e = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        a = tf.nn.softmax(e, axis=1)
        c = a * hidden_states
        c = tf.reduce_sum(c, axis=1)
        return c, a
