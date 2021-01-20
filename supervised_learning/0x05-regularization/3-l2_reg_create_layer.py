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

    λ = lambtha

    raw_layer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_regularize = tf.contrib.layers.l2_regularizer(scale=λ, scope=None)

    output_tensor = tf.layers.Dense(units=n,
                                    activation=activation,
                                    kernel_initializer=raw_layer,
                                    kernel_regularizer=new_regularize,
                                    name="layer")

    return(output_tensor(prev))
