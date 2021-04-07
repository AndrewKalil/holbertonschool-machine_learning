#!/usr/bin/env python3
""" Markov chain: baum welch algorithm """
import numpy as np

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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    performs the Baum-Welch algorithm for a hidden markov model
    :param Observations: numpy.ndarray of shape (T,)
        that contains the index of the observation
        T is the number of observations
    :param Transition: numpy.ndarray of shape (M, M)
        that contains the initialized transition probabilities
        M is the number of hidden states
    :param Emission: numpy.ndarray of shape (M, N)
        that contains the initialized emission probabilities
        N is the number of output states
    :param Initial: numpy.ndarray of shape (M, 1)
        that contains the initialized starting probabilities
    :param iterations: number of times expectation-maximization
        should be performed
    :return: the converged Transition, Emission, or None, None on failure
    """
    # type and len(dim) conditions
    if not isinstance(Observations, np.ndarray) \
            or len(Observations.shape) != 1:
        return None, None

    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    if not isinstance(Transition, np.ndarray) or len(Transition.shape) != 2:
        return None, None

    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None

    # dim conditions
    T = Observations.shape[0]
    N, M = Emission.shape

    if Transition.shape[0] != N or Transition.shape[1] != N:
        return None, None

    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None, None

    # stochastic
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.sum(Transition, axis=1).all():
        return None, None
    if not np.sum(Initial) == 1:
        return None, None

    for n in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = np.matmul(alpha[:, t].T, Transition)
            b = Emission[:, Observations[t + 1]].T
            c = beta[:, t + 1]
            denominator = np.matmul(a * b, c)

            for i in range(N):
                a = alpha[i, t]
                b = Transition[i]
                c = Emission[:, Observations[t + 1]].T
                d = beta[:, t + 1].T
                numerator = a * b * c * d
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)

        # TRANSITION CALCULATION
        num = np.sum(xi, 2)
        den = np.sum(gamma, axis=1).reshape((-1, 1))
        Transition = num / den

        # EMISSION CALCULATION
        # add additional T'th element in gamma
        xi_sum = np.sum(xi[:, :, T - 2], axis=0)
        xi_sum = xi_sum.reshape((-1, 1))
        gamma = np.hstack((gamma, xi_sum))

        denominator = np.sum(gamma, axis=1)
        denominator = denominator.reshape((-1, 1))

        for i in range(M):
            gamma_i = gamma[:, Observations == i]
            Emission[:, i] = np.sum(gamma_i, axis=1)

        Emission = Emission / denominator

    return Transition, Emission
