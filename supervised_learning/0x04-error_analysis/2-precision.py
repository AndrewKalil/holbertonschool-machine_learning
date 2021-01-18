#!/usr/bin/env python3
"""Create confusion"""
import numpy as np


def precision(confusion):
    """
    confusion is a confusion numpy.ndarray of shape (classes, classes)
     where row indices represent the correct labels and column indices
     represent the predicted labels
        classes is the number of classes
    """
    precision = []
    i = 0
    for row in confusion:
        positive = row[i]
        column = confusion.sum(axis=0)
        # print(positive, column)
        precision.append(positive / column[i])
        i = i + 1

    return np.array(precision)
