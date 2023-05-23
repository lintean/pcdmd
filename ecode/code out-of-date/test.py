import numpy as np
import pandas as pd

a = np.array([[1, 2, 1, 2, 3, 5, 6, 2]])

b = np.array([
    [1, 2, 1, 2, 3, 5, 6, 2],
    [1, 2, 1, 2, 3, 5, 6, 2],
    [2, 3, 3, 2, 1, 2, 1, 1],
    [2, 2, 1, 1, 1, 2, 4, 4]
])

c = np.array([
    [1, 2, 1, 2, 3, 5, 6, 2],
    [2, 1, 1, 2, 1, 2, 3, 3],
    [2, 3, 3, 2, 1, 2, 1, 1],
    [2, 2, 1, 1, 1, 2, 4, 4]
])

print("b shape:")
print(b.shape)
print("c shape:")
print(c.shape)

def matrix_matrix(arr, brr):
    sum = arr.dot(brr.T)
    # print(arr * brr)
    # print(sum)
    sum = sum.diagonal()
    # print(sum)
    sum = sum / ((np.sqrt(np.sum(arr*arr, axis=1))) * np.sqrt(np.sum(brr*brr, axis=1)))
    return sum

print(matrix_matrix(b, c))