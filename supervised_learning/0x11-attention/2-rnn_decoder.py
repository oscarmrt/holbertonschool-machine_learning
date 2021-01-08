#!/usr/bin/env python3
"""Class RNNDecoder that inherits from tensorflow.keras.layers.Layer
to decode for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """class RNNDecoder"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor"""
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Public instance method that returns y, s"""
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], -1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return (y, state)
