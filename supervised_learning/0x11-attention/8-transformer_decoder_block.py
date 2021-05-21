#!/usr/bin/env python3
"""
Class DecoderBlock
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ Decoder Block Class
    creates a decoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class Constructor
        """
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
        """ Public Instance Method
        calling transformers
        """
        attn_1, weights_b1 = self.mha1(x, x, x, look_ahead_mask)
        attn_1 = self.dropout1(attn_1, training=training)
        output1 = self.layernorm1(x + attn_1)

        attn_2, weights_b2 = self.mha2(output1,
                                       encoder_output,
                                       encoder_output,
                                       padding_mask)
        attn_2 = self.dropout2(attn_2, training=training)

        output2 = self.layernorm2(attn_2 + output1)
        ffn_out = self.dense_hidden(output2)
        ffn_out = self.dense_output(ffn_out)
        ffn_out = self.dropout3(ffn_out, training=training)
        output3 = self.layernorm3(ffn_out + output2)

        return output3
