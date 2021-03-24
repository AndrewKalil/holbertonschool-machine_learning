#!/usr/bin/env python3

import numpy as np
pca = __import__('0-pca').pca

np.random.seed(0)
a = np.random.normal(size=50)
# print("before", a, end="\n\n")
b = np.random.normal(size=50)
# print("before", b, end="\n\n")
c = np.random.normal(size=50)
# print("before", c, end="\n\n")
d = 2 * a
# print(d, end="\n\n")
e = -5 * b
# print(e, end="\n\n")
f = 10 * c
# print(f, end="\n\n")

X = np.array([a, b, c, d, e, f]).T
m = X.shape[0]
X_m = X - np.mean(X, axis=0)
# u, s, vh = np.linalg.svd(X_m)
# print(u, end='\n\n')
# print(vh, end='\n\n')
# print(u, end='\n\n')
W = pca(X_m)
T = np.matmul(X_m, W)
print(T)
X_t = np.matmul(T, W.T)
print(np.sum(np.square(X_m - X_t)) / m)
