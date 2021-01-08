#!/usr/bin/env python3
"""Class DecoderBlock"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """Class DecoderBlock"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Public instance method that returns a tensor of
        shape (batch, target_seq_len, dm) containing the block’s
        output"""
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        output1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(output1, encoder_output,
                                               encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        output2 = self.layernorm2(attn2 + output1)
        ffn = tf.keras.Sequential([self.dense_hidden, self.dense_output])
        ffn_output = ffn(output2)
        ffn_output = self.dropout3(ffn_output, training=training)
        output3 = self.layernorm3(ffn_output + output2)
        return output3
