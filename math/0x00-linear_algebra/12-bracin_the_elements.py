#!/usr/bin/env python3
""" 0x00. Linear Algebra """


def np_elementwise(mat1, mat2):
    """ inner element operations """

    add = mat1+mat2
    sub = mat1-mat2
    mul = mat1*mat2
    div = mat1/mat2
    return (add, sub, mul, div)
