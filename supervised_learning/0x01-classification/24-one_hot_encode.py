#!/usr/bin/env python3
"""One Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    emptyMatrix = np.zeros((len(Y), classes))

    if Y is not None:
        for i in range(classes):
            emptyMatrix[i][Y[i]] = 1

        return emptyMatrix.T
    else:
        return None
