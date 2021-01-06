#!/usr/bin/env python3
""" Loss """
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network

    Args:
        loss: the loss of the network’s prediction
        alpha: the learning rate
    """
    opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return opt
