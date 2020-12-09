#!/usr/bin/env python3
""" Our life is the sum total of all the decisions we make
    every day, and those decisions are determined by our priorities  """


def summation_i_squared(n):
    """ making a summation function """

    if type(n) is not int or n < 1:
        return None
    if n == 1:
        return 1
    else:
        return int((n*((n+1)*(2*n+1))) / 6)
