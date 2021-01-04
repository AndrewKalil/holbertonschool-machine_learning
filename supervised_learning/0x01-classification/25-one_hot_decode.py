#!/usr/bin/env python3
"""One Hot Decode"""
import numpy as np


def one_hot_decode(one_hot):
    """ Function for one hot encoding"""
    if type(one_hot) is not np.ndarray\
       or len(one_hot.shape) != 2:
        return None
    return np.array([np.where(i == 1)[0][0]
                     for i in one_hot.T])
