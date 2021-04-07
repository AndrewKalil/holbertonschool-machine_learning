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


def viterbi(Observation, Emission, Transition, Initial):
    """
    Function that  calculates the most likely sequence of hidden states for a
    hidden markov model:
    Arguments
    ---------
    - Observation : numpy.ndarray
                    Array of shape (T,) that contains the index of the obs
                T : int
                    Number of observations
    - Emission    : numpy.ndarray
                    Array of shape (N, M) containing the emission probability
                    of a specific observation given a hidden state
                    Emission[i, j] is the probability of observing j given the
                    hidden state i
                N : int
                    Number of hidden states
                M : int
                    Number of all possible observations
    - Transition  : numpy.ndarray
                    2D array of shape (N, N) containing the transition probs
                    Transition[i, j] is the prob of transitioning from the
                    hidden state i to j
    - Initial     : numpy.ndarray
                    Array of shape (N, 1) containing the probability of
                    starting in a particular hidden state
    Returns
    -------
    path, P, or None, None on failure
    - path        : list
                    list of length T containing the most likely sequence of
                    hidden states
    - p           : float
                    The probability of obtaining the path sequence
    """

    N = Emission.shape[0]

    try:
        if (Observation.ndim != 1 or Emission.ndim != 2):
            return None, None

        if (Transition.shape != (N, N) or Initial.shape != (N, 1)):
            return None, None

        if (not np.isclose(np.sum(Emission, axis=1), 1).all()):
            return None, None

        if (not np.isclose(np.sum(Transition, axis=1), 1).all()):
            return None, None

        if (not np.isclose(np.sum(Initial, axis=0), 1).all()):
            return None, None

        T = Observation.shape[0]
        F = np.zeros((N, T))
        prev = np.zeros((N, T))

        # Initilaize the tracking tables from first observation
        F[:, 0] = Initial.T * Emission[:, Observation[0]]
        prev[:, 0] = 0

        # Iterate throught the observations updating the tracking tables
        for i, obs in enumerate(Observation):
            if i != 0:
                F[:, i] = np.max(F[:, i - 1] * Transition.T *
                                 Emission[np.newaxis, :, obs].T, 1)
                prev[:, i] = np.argmax(F[:, i - 1] * Transition.T, 1)

        # Build the output, optimal model trajectory (path)
        path = T * [1]
        path[-1] = np.argmax(F[:, T - 1])
        for i in reversed(range(1, T)):
            path[i - 1] = int(prev[path[i], i])

        # calculate the probability of obtaining the path sequence
        P = np.amin(np.amax(F, axis=0))

        return path, P

    except BaseException:
        return None, None


def backward(Observation, Emission, Transition, Initial):
    """
    Function that performs the backward algorithm for a hidden markov model
    Arguments
    ---------
    - Observation : numpy.ndarray
                    Array of shape (T,) that contains the index of the obs
                T : int
                    Number of observations
    - Emission    : numpy.ndarray
                    Array of shape (N, M) containing the emission probability
                    of a specific observation given a hidden state
                    Emission[i, j] is the probability of observing j given the
                    hidden state i
                N : int
                    Number of hidden states
                M : int
                    Number of all possible observations
    - Transition  : numpy.ndarray
                    2D array of shape (N, N) containing the transition probs
                    Transition[i, j] is the prob of transitioning from the
                    hidden state i to j
    - Initial     : numpy.ndarray
                    Array of shape (N, 1) containing the probability of
                    starting in a particular hidden state
    Returns
    -------
    P, F, or None, None on failure
    - P           : float
                    likelihood of the observations given the model
    - F:          : Numpy.ndarray
                    Array of shape (N, T) containing the backward path
                    probabilities B[i, j] is the probability of being in
                    hidden state i at time j in the future observations
    """

    try:
        N = Emission.shape[0]

        if (Observation.ndim != 1) or (Emission.ndim != 2):
            return None, None

        if (Transition.shape != (N, N)) or (Initial.shape != (N, 1)):
            return None, None

        if (not np.isclose(np.sum(Emission, axis=1), 1).all()):
            return None, None

        if (not np.isclose(np.sum(Transition, axis=1), 1).all()):
            return None, None

        if (not np.isclose(np.sum(Initial, axis=0), 1).all()):
            return None, None

        T = Observation.shape[0]
        B = np.ones((N, T))

        for obs in reversed(range(T - 1)):
            for h_state in range(N):
                B[h_state, obs] = (np.sum(B[:, obs + 1] *
                                          Transition[h_state, :] *
                                          Emission[:, Observation[obs + 1]]))

        P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])

        return P, B

    except BaseException:
        return None, None


def baum_welch(Observations, N, M, Transition=None,
               Emission=None, Initial=None):
    """
    Function that performs the Baum - Welch algorithm for a hidden
    markov model:
    Arguments
    ---------
    - Observation     : numpy.ndarray
                        Array of shape(T,) that contains the index of the
                        observation
                    T : int
                        Number of observations
    - N               : int
                        Number of hidden states
    - M               : int
                        Number of possible observations
    - Transition      : numpy.ndarray
                        Initialized transition probs, defaulted to None
    - Emission        : numpy.ndarray
                        Initialized emission probs, defaulted to None
    - Initial         : Initiallized starting probabilities, defaulted to None
    If Transition, Emission, or Initial is None, initialize the probabilities
    as being a uniform distribution
    Returns
    -------
    The converged Transition, Emission, or None, None on failure
    """

    try:

        # 1. Type validations
        if (not isinstance(Observations, np.ndarray)) or (
                not isinstance(M, int)) or (not isinstance(N, int)):
            return None, None

        # 2. Dim validations
        if (Observations.ndim != 1):
            return None, None

        T = Observations.shape[0]
        if (not isinstance(T, int)):
            return None, None

        # 3. None validation
        if (Transition is None):
            pass

        if (Emission is None):
            pass

        if (Initial is None):
            pass

        # https://tinyurl.com/yc4waatv   (www.adeveloperdiary.com)
        # https://tinyurl.com/yd4skvvy   (github)

        M = Transition.shape[0]
        n_iter = 100

        for n in range(n_iter):
            alpha = forward(Observations, Transition, Emission, Initial)
            beta = backward(Observations, Transition, Emission, Initial)

            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                a = alpha[t, :].T
                b = Transition
                c = Observations[t + 1]
                d = Emission[:, c].T
                e = beta[t + 1, :]
                denom = np.dot(np.dot(a, b) * d, e)

                for i in range(M):
                    a1 = alpha[t, i]
                    b1 = Transition[i, :]
                    c1 = Observations[t + 1]
                    d1 = Emission[:, c1].T
                    e1 = beta[t + 1, :].T
                    numerator = a1 * b1 * d1 * e1

                    xi[i, :, t] = numerator / denom

                if (n):
                    pass

            gamma = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

            # Add additional T'th element in gamma
            gamma = np.hstack(
                (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

            K = Emission.shape[1]
            denominator = np.sum(gamma, axis=1)
            for x in range(K):
                Emission[:, x] = np.sum(gamma[:, Observations == x], axis=1)

            b = np.divide(Emission, denominator.reshape((-1, 1)))

        return a, b

    except BaseException:
        return None, None
