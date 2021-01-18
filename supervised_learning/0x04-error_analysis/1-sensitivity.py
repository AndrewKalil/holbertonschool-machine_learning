#!/usr/bin/env python3
"""Create confusion"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
         where row indices represent the correct labels and column indices
         represent the predicted labels
            classes is the number of classes
    """
    sensitivity = []
    # print(confusion)
    i = 0
    for row in confusion:
        positive = row[i]
        false_positive = sum(row)
        # print(positive, false_positive)
        sensitivity.append(positive / false_positive)
        i = i + 1

    return np.array(sensitivity)
