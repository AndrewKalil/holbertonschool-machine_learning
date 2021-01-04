#!/usr/bin/env python3
"""One Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """ Function for one hot encoding"""
    if Y is None\
       or type(Y) is not np.ndarray\
       or type(classes) is not int:
        return None
    try:
        matrix = np.zeros((len(Y), classes))
        matrix[np.arange(len(Y)), Y] = 1
        return matrix.T
    except Exception:
        return None
