#!/usr/bin/env python3
""" Regulization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization

    Args:
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation function that should be used on the layer
        lambtha is the L2 regularization parameter
    """
    mode = "FAN_AVG"
    regulizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    initializer = tf.contrib.layers.variance_scaling_initializer(mode=mode)
    layer = tf.layers.Dense(kernel_initializer=initializer,
                            kernel_regularizer=regulizer,
                            units=n,
                            activation=activation)
    return layer(prev)
