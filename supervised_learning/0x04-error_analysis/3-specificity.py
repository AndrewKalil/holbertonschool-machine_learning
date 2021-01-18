#!/usr/bin/env python3
"""Create confusion"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in a confusion matrix

    confusion is a confusion numpy.ndarray of shape (classes, classes)
     where row indices represent the correct labels and column indices
     represent the predicted labels
        classes is the number of classes
    """
    # print(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    # print(confusion.sum(axis=0), np.diag(confusion), FP)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    # print(confusion.sum(axis=1), np.diag(confusion), FN)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    return (TN / (TN + FP))
