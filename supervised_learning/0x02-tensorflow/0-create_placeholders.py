#!/usr/bin/env python3
"""placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """returns two placeholders, x and y, for the neural network

    Args:
        nx: number of feature columns in our data
        classes: number of classes in our classifier
    """
    x = tf.placeholder(float, shape=[None, nx], name='x')
    y = tf.placeholder(float, shape=[None, classes], name='y')

    return (x, y)