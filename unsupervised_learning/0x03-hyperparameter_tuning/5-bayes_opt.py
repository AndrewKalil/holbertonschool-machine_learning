#!/usr/bin/env python3
""" Bayesian Optimization """
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """ Bayesian Optimization Class """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Instantiation method

        Args:
            f is the black-box function to be optimized
            X_init is a numpy.ndarray of shape (t, 1) representing
              the inputs already sampled with the black-box function
            Y_init is a numpy.ndarray of shape (t, 1) representing
              the outputs of the black-box function for each input in X_init
            t is the number of initial samples
            bounds is a tuple of (min, max) representing the bounds of
              the space in which to look for the optimal point
            ac_samples is the number of samples that should be analyzed
              during acquisition
            l is the length parameter for the kernel
            sigma_f is the standard deviation given to the output of the
              black-box function
            xsi is the exploration-exploitation factor for acquisition
            minimize is a bool determining whether optimization should be
              performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.zeros((ac_samples, 1))
        self.X_s = np.linspace(start=bounds[0],
                               stop=bounds[1],
                               num=ac_samples,
                               endpoint=True)
        self.X_s = self.X_s.reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ calculates the next best sample location """
        m_sample, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            sam = np.min(self.gp.Y)
            imp = sam - m_sample - self.xsi
        else:
            sam = np.max(self.gp.Y)
            imp = m_sample - sam - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_nest = self.X_s[np.argmax(EI)]
        return X_nest, EI

    def optimize(self, iterations=100):
        """optimizes the black-box function

        Args:
            iterations is the maximum number of iterations to perform
        """
        X_opt = 0
        Y_opt = 0

        for _ in range(iterations):
            # Find the next best sample
            X_next = self.acquisition()[0]

            # if X_next already sampled in gp.X, ommit
            if (X_next in self.gp.X):
                break

            else:
                # get Y_next, evaluate X_next is black box function
                Y_next = self.f(X_next)

                # updates a GP
                self.gp.update(X_next, Y_next)

                # if minimizing save the least otherwise save the largest
                if (Y_next < Y_opt) and (self.minimize):
                    X_opt, Y_opt = X_next, Y_next

                if not self.minimize and Y_next > Y_opt:
                    X_opt, Y_opt = X_next, Y_next

        # removing last element
        self.gp.X = self.gp.X[:-1]

        return X_opt, Y_opt
