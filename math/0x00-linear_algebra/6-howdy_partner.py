#!/usr/bin/env python3
""" """

def cat_arrays(arr1, arr2):
    """ """
    
    if len(arr1) > 0:
        new_list = list(arr1)
        if len(arr2) > 0:
            for i in range(len(arr2)):
                new_list.append(arr2[i])
            return new_list
            