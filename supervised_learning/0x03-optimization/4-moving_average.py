#!/usr/bin/env python3
""" Description """
import numpy as np
import tensorflow as tf


def moving_average(data, beta):
    """calculates the weighted moving average of a data set

    Args:
        data:  list of data to calculate the moving average of
        beta: weight used for the moving average
    """
    newData = []
    value = 0
    for i in range(len(data)):
        value = beta * value + (1 - beta) * data[i]
        newData.append(value/(1 - beta ** (i + 1)))

    return newData
