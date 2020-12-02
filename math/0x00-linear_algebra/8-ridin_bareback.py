#!/usr/bin/env python3
""" 0x00. Linear Algebra """


def matrix_shape(matrix):
    """ Returns shape of matrix """

    if matrix:
        shape = [len(matrix)]
        while type(matrix[0]) == list:
            shape.append(len(matrix[0]))
            matrix = matrix[0]
        return shape
    else:
        return [0]


def mat_mul(mat1, mat2):
    """ multiplies a matrix """

    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    new_mat = []
    if shape1[1] == shape2[0]:
        for i in range(shape1[0]):
            new_row = []
            for j in range(shape2[1]):
                element = 0
                for k in range(shape1[1]):
                    addend = mat1[i][k] * mat2[k][j]
                    element += addend
                new_row.append(element)
            new_mat.append(new_row)
        return new_mat
    else:
        return None
