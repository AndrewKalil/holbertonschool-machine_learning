#!/usr/bin/env python3
""" Poisson Distribution """


class Normal:
    """represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """Instantiation

        Args:
            data (list): data to be used to estimate the
                distribution. Defaults to None.
            mean (int): mean of the distribution. Defaults
                to 0..
            stddev (float): standard deviation of the distribution.
                Defaults to 1..
        """
        self.e = 2.7182818285
        self.pi = 3.1415926536

        if data is not None:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            new_list = []
            for i in data:
                new_list.append((i-self.mean)**2)
            self.stddev = (sum(new_list) / len(data)) ** 0.5
        else:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = mean
            self.stddev = stddev

    def z_score(self, x):
        """Calculates the z score

        Args:
            x (int): x score

        Returns:
            float: z score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates x score

        Args:
            z (int): z score

        Returns:
            float: x score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Probability density function

        Args:
            x (int/float): x value
        """
        return (self.e ** -((x-self.mean)**2 / (2*self.stddev**2)))\
            / (self.stddev * (2*self.pi)**.5)

    def cdf(self, x):
        """Cumulative density function

        Args:
            x (int/float): x value
        """
        arg = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / 3.1415926536 ** 0.5) * \
              (arg - (arg ** 3) / 3 + (arg ** 5) / 10 -
               (arg ** 7) / 42 + (arg ** 9) / 216)
        return (1/2) * (1 + erf)
