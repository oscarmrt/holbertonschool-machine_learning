#!/usr/bin/env python3
"""Class RNNEncoder that inherits from tensorflow.keras.layers.Layer
to encode for machine translation"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """Class RNNEncoder"""
    def __init__(self, vocab, embedding, units, batch):
        """Class constructor"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(self.units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Public instance method that initializes the hidden
        states for the RNN cell to a tensor of zeros"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Public instance method that returns outputs, hidden"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
