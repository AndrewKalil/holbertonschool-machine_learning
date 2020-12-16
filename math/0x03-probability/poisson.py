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


class Poisson:
    """ Represents the Posisson distribution """

    def __init__(self, data=None, lambtha=1):
        """Instantiation method

        Args:
            data (list): data to be used to estimate the distribution.
                Defaults to None.
            lambtha (int, optional): expected number of occurences in
                a given time frame. Defaults to 1.
        """
        self.e = 2.7182818285

        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

    def pmf(self, k):
        """probability mass function

        Args:
            k (int): number of successes
        """
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        summation = ((self.e**-self.lambtha) *
                     (self.lambtha**k)) / factorial(k)
        return summation

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”

        Args:
            k (int):  the number of “successes”
        """
        if k < 0:
            return 0
        if type(k) is not int:
            k = int(k)
        summation = 0
        for i in range(0, k+1):
            summation = summation + self.pmf(i)
        return summation
