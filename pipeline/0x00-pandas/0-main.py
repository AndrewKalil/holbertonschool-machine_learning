#!/usr/bin/env python3

import numpy as np
from_numpy = __import__('0-from_numpy').from_numpy

np.random.seed(0)
A = np.random.randn(5, 8)
# print("A: ", A)
print(from_numpy(A))
B = np.random.randn(9, 3)
# print("B: ", B)
print(from_numpy(B))
