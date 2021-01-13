#!/usr/bin/env python3
""" Description """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """updates a variable in place using the Adam optimization algorithm

    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta1 is the weight used for the first moment
        beta2 is the weight used for the second moment
        epsilon is a small number to avoid division by zero
    """
    α = alpha
    β1 = beta1
    β2 = beta2
    ε = epsilon

    train = tf.train.AdamOptimizer(learning_rate=α,
                                   beta1=β1,
                                   beta2=β2,
                                   epsilon=ε)

    return train.minimize(loss)
