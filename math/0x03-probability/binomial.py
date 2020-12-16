#!/usr/bin/env python3
""" Poisson Distribution """


def factorial(n):
    """calculates the factrial of a number

    Args:
        n (int): number to find the factorial for

    Returns:
        int: factorial of n
    """
    if n == 0:
        return 1

    fact = 1

    for i in range(1, n+1):
        fact = fact * i
    return fact


class Binomial:
    """Binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """Instantiation method

        Args:
            data (list):  data to be used to estimate the distribution.
                Defaults to None.
            n (int): number of Bernoulli trials. Defaults to 1.
            p (float): probability of a “success”. Defaults to 0.5.
        """
        if data is not None:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data)/len(data))
            new_data = [(x - mean) ** 2 for x in data]
            variance = sum(new_data) / len(data)
            p = 1 - variance / mean
            if ((mean / p) - (mean // p)) >= 0.5:
                self.n = 1 + int(mean / p)
            else:
                self.n = int(mean / p)
            self.p = float(mean / self.n)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if (p <= 0) or (p >= 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
            self.n = int(n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”

        Args:
            k (int): number of number of successes
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0

        comb = factorial(self.n) / (factorial(self.n-k) * factorial(k))
        return comb * self.p**k * ((1-self.p)**(self.n-k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”

        Args:
            k (int): number of “successes”
        """
        cdf = 0

        for i in range(k+1):
            cdf += self.pmf(i)
        return cdf
