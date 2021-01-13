#!/usr/bin/env python3
""" Description """
import numpy as np
import tensorflow as tf


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
            d is the number of data points
            nx is the number of features
        m: numpy.ndarray of shape (nx,) that contains the mean of
         all features of X
        s: numpy.ndarray of shape (nx,) that contains the standard
         deviation of all features of X
    """
    Z = (X - m) / s
    return Z