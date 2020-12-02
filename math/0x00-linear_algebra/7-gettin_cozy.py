#!/usr/bin/env python3
""" """


def cat_matrices2D(mat1, mat2, axis=0):
    """ """

    if axis == 0 and len(mat1[0]) != len(mat2[0]):
        return None
    if axis == 1 and len(mat1) != len(mat2):
        return None
    new_mat = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j])
        if axis == 1:
            for j in range(len(mat2[0])):
                row.append(mat2[i][j])
        new_mat.append(row)

    if axis == 0:
        for i in range(len(mat2)):
            row = []
            for j in range(len(mat2[0])):
                row.append(mat2[i][j])
            new_mat.append(row)

    return new_mat
