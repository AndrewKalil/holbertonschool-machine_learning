#!/usr/bin/env python3
""" Description """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev is the activated output of the previous layer
        n is the number of nodes in the layer to be created
        activation is the activation function that should be used on the output
         of the layer
         used for batch normalization
        epsilon is a small number used to avoid division by zero
    """
    initializer = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    model = tf.layers.Dense(units=n,
                            activation=None,
                            kernel_initializer=initializer,
                            name='layer')

    mean, variance = tf.nn.moments(model(prev), axes=0, keep_dims=True)

    beta = tf.Variable(tf.constant(0.0, shape=[n]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]),
                        name='gamma', trainable=True)
    adjusted = tf.nn.batch_normalization(model(prev), mean, variance,
                                         offset=beta, scale=gamma,
                                         variance_epsilon=1e-8)
    if activation is None:
        return model(prev)
    return activation(adjusted)
