#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves a model’s weights
    :param network: model whose weights should be saved
    :param filename: path of the file that the model should be saved to
    :param save_format:  format in which the weights should be saved
    :return: None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
     loads a model’s weights
     :param network: model to which the weights should be loaded
    :param filename:  path of the file that the model should be loaded from
    :return: None
    """
    network.load_weights(filename)
    return None
