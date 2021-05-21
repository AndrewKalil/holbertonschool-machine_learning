#!/usr/bin/env python3
"""
Function that calcuylates the scaled dot product attention
"""


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    calculates scaled dot product attention
    """

    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    scald_qk = tf.cast(tf.shape(K)[-1], tf.float32)
    scald_atten = matmul_qk / tf.math.sqrt(scald_qk)

    if mask is not None:
        scald_atten += (mask * -1e9)
    weights = tf.nn.softmax(scald_atten, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
