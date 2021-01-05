#!/usr/bin/env python3
"""Creates Layers"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates layers
    Args:
        prev: the tensor output of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function that the layer should use
    """
    raw_layer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    output_tensor = tf.layers.Dense(units=n,
                                    activation=activation,
                                    kernel_initializer=raw_layer,
                                    name="layer")
    return(output_tensor(prev))
