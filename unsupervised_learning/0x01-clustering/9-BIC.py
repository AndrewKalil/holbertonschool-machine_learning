#!/usr/bin/env python3

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that performs the expectation maximization for a GMM:
    Args:
		X: numpy.ndarray -> Array of shape (n, d) with the data
    	kmin: int -> positive int with the minimum number
            of clusters to check for (inclusive)
    	kmax: int -> positive int with the maximum number
            of clusters to check for (inclusive)
    	iterations: int -> positive int with the maximum number
            of iterations for the algorithm
    	tol: float -> non-negative float with the tolerance
            of the log likelihood, used to
            determine early stopping i.e. if the
            difference is less than or equal to
            tol you should stop the algorithm
    	verbose: bool -> boolean that determines if you should
            print information about the algorithm
            If True, print Log Likelihood after {i} iterations: {l}
            every 10 iterations and after the last iteration
                {i} is the number of iterations of the EM algorithm
                 {l} is the log likelihood
            You should use:
            initialize = __import__('4-initialize').initialize
            expectation = __import__('6-expectation').expectation
            maximization = __import__('7-maximization').maximization
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S,  _, log_l = expectation_maximization(X, k, iterations, tol,
                                                       verbose)
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)

        # Formula pf paramaters: https://bit.ly/33Cw8lH
        # code based on gaussian mixture source code n_parameters source code
        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        # Formula for this task BIC = p * ln(n) - 2 * l
        bic = p * np.log(n) - 2 * log_l
        bic_val.append(bic)

    bic_val = np.array(bic_val)
    logl_val = np.array(logl_val)
    best_val = np.argmin(bic_val)

    k_best = k_best[best_val]
    best_res = best_res[best_val]

    return k_best, best_res, logl_val, bic_val