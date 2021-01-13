#!/usr/bin/env python3
""" Description """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ updates a variable using the RMSProp optimization algorithm

    Args:
        loss is the loss of the network
        alpha is the learning rate
        beta2 is the RMSProp weight
        epsilon is a small number to avoid division by zero
    """
    α = alpha
    β2 = beta2
    ε = epsilon

    train = tf.train.RMSPropOptimizer(learning_rate=α, decay=β2, epsilon=ε)

    return train.minimize(loss)
