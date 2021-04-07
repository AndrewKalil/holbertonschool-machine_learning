#!/usr/bin/env python3
""" Regular Markov chain """
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


def regular(P):
    """determines the steady state probabilities of a regular markov chain

    Args:
            P is a is a square 2D numpy.ndarray of shape (n, n) representing
              the transition matrix
                P[i, j] is the probability of transitioning from state i to
                  state j
                n is the number of states in the markov chain
    """
    np.warnings.filterwarnings('ignore')
    # Avoid this warning: Line 92.  np.linalg.lstsq(a, b)[0]

    if (not isinstance(P, np.ndarray)):
        return None

    if (P.ndim != 2):
        return None

    n = P.shape[0]
    if (P.shape != (n, n)):
        return None

    if ((np.sum(P) / n) != 1):
        return None

    if ((P > 0).all()):  # checks to see if all elements of P are posistive
        a = np.eye(n) - P
        a = np.vstack((a.T, np.ones(n)))
        b = np.matrix([0] * n + [1]).T
        regular = np.linalg.lstsq(a, b)[0]
        return regular.T

    return None
