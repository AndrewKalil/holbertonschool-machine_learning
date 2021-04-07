#!/usr/bin/env python3
""" Makov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain being in a
      particular state after a specified number of iterations

    Args:
            P is a square 2D numpy.ndarray of shape (n, n)
              representing the transition matrix
                    P[i, j] is the probability of transitioning from state
                      i to state j
                    n is the number of states in the markov chain
            s is a numpy.ndarray of shape (1, n) representing the
              probability of starting in each state
            t is the number of iterations that the markov chain has
              been through
    """
    if (not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray)):
        return None

    if (not isinstance(t, int)):
        return None

    if ((P.ndim != 2) or (s.ndim != 2) or (t < 1)):
        return None

    n = P.shape[0]
    if (P.shape != (n, n)) or (s.shape != (1, n)):
        return None

    while (t > 0):
        s = np.matmul(s, P)
        t -= 1

    return s
