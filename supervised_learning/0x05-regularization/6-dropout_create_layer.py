#!/usr/bin/env python3
""" Regulization """
import numpy as np
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout
    Args:
        prev is a tensor containing the output of the previous layer
        n is the number of nodes the new layer should contain
        activation is the activation funct that should be used on the layer
        keep_prob is the probability that a node will be kept
    Returns:
        the output of the new layer
    """

    raw_layer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout_layer = tf.layers.Dropout(keep_prob)

    output_tensor = tf.layers.Dense(units=n,
                                    activation=activation,
                                    kernel_regularizer=dropout_layer,
                                    kernel_initializer=raw_layer)

    applied_layer = output_tensor(prev)
    return(applied_layer)
