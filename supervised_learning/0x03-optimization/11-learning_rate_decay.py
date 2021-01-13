#!/usr/bin/env python3
""" Description """
import numpy as np
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay in numpy

    Args:
        alpha is the original learning rate
        decay_rate is the weight used to determine the rate at which alpha
         will decay
        global_step is the number of passes of gradient descent
         that have elapsed
        decay_step is the number of passes of gradient descent
         that should occur before alpha is decayed further
    """
    α = alpha
    dr = decay_rate

    decayed_learning_rate = α / (1 + dr * int(global_step / decay_step))
    return (decayed_learning_rate)
