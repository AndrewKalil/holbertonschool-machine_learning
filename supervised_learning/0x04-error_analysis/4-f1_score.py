#!/usr/bin/env python3
"""Create confusion"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """calculates the F1 score of a confusion matrix

    Args:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
         where row indices represent the correct labels and column indices
         represent the predicted labels
            classes is the number of classes
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)

    return 2 * (prec * sens)/(prec + sens)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)

    # Precision or positive predictive value
    PPV = TP / (TP + FP)

    F1 = 2 / ((1 / TPR) + (1 / PPV))
    return F1
