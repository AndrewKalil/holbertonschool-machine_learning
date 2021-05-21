#!/usr/bin/env python3
"""
Class EncoderBlock
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """create encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        class Constructor
        """
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)

        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        calling transformers
        """
        attnOutput, _ = self.mha(x, x, x, mask)
        attnOutput = self.dropout1(attnOutput, training=training)
        output_3 = self.layernorm1(x + attnOutput)
        output_2 = self.dense_hidden(output_3)
        output_1 = self.dense_output(output_2)
        output_0 = self.dropout2(output_1, training=training)
        output = self.layernorm2(output_3 + output_0)

        return output
