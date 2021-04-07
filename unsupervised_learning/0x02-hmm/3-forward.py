#!/usr/bin/env python3
""" Absorbing Markov chain: forward algorithm """
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


def absorbing(P):
    """determines if a markov chain is absorbing

    Args:
        P is a is a square 2D numpy.ndarray of shape
          (n, n) representing the standard transition matrix
            P[i, j] is the probability of transitioning from state
              i to state j
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

    # P is an identity matrix
    identity = np.eye(n)
    if (np.equal(P, identity).all()):
        return True

    abs = np.zeros(n)
    for i in range(n):
        if P[i][i] == 1:
            abs[i] = 1

    prev = np.zeros(n)
    while (not np.array_equal(abs, prev) and abs.sum() != n):
        prev = abs.copy()
        for absorbed in P[:, np.nonzero(abs)[0]].T:
            abs[np.nonzero(absorbed)] = 1
    if (abs.sum() == n):
        return True
    return False


def forward(Observation, Emission, Transition, Initial):
    """
    Function that performs the forward algorithm for a hidden markov model:
    Args:
    - Observation       numpy.ndarray       Array of shape (T,) that contains
                                            the index of the observation
                T       int                 Number of observations
    - Emission          numpy.ndarray       Array of shape (N, M) containing
                                            the emission probability of a
                                            specific observation given a
                                            hidden state
                                            - Emission[i, j] is the probability
                                            of observing j given the hidden
                                            state i
            - N         int                 Number of hidden states
            - M         int                 Number of all possible observations
    - Transition        numpy.ndarray       2D array of shape (N, N) containing
                                            the transition probabilities
                                            - Transition[i, j] is the prob
                                            of transitioning from the hidden
                                            state i to j
    - Initial           numpy.ndarray       Array of shape (N, 1) containing
                                            the probability of starting in a
                                            particular hidden state
    Returns: P, F, or None, None on failure
    - P:                                    likelihood of the observations
                                            given the model
    - F:                Numpy.ndarray       Array of shape (N, T) containing
                                            the forward path probabilities
                                            F[i, j] is the probability of
                                            being in hidden state i at time j
                                            given the previous observations
    """
    try:
        if (not isinstance(Observation, np.ndarray)) or (
                not isinstance(Emission, np.ndarray)) or (
                not isinstance(Transition, np.ndarray)) or (
                not isinstance(Initial, np.ndarray)):
            return None, None

        # 2. Dim validations
        if (Observation.ndim != 1) or (
                Emission.ndim != 2) or (
                Transition.ndim != 2) or (
                Initial.ndim != 2):
            return None, None

        # 3. Structure validations
        if (not np.sum(Emission, axis=1).all() == 1) or (
                not np.sum(Transition, axis=1).all() == 1) or (
                not np.sum(Initial).all() == 1):
            return None, None

        T = Observation.shape[0]
        N = Emission.shape[0]

        forward = np.zeros((N, T))
        forward[:, 0] = Initial.T * Emission[:, Observation[0]]

        for t in range(1, T):
            for j in range(N):
                forward[j, t] = (forward[:, t - 1].dot(Transition[:, j])
                                 * Emission[j, Observation[t]])

        likelihood = np.sum(forward[:, t])
        return likelihood, forward

    except BaseException:
        return None, None
