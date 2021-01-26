#!/usr/bin/env python3
""" Keras """
import tensorflow.keras as K


def save_model(network, filename):
    """
    saves an entire model
    :param network: model to save
    :param filename: path of the file that the model should be saved to
    :return: None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    loads an entire model
    :param filename:  path of the file that the model should be loaded from
    :return: loaded model
    """
    network = K.models.load_model(filename)
    return network
