#!/usr/bin/env python3
"""Class EncoderBlock that inherits from tensorflow.keras.layers.Layer to
create an encoder block for a transformer"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """Class EncoderBlock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Public instance method that returns a tersor of
        shape (batch, input_seq_len, dm) containing the block’s
        output"""
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        output1 = self.layernorm1(x + attn_output)
        ffn = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        ffn_output = ffn(output1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)
        return output2
