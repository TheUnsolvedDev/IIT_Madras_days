import numpy as np


def trial(array1, array2, N):
    new_array = np.zeros_like(array1)
    for i in range(len(array1)):
        new_array[i] = array1[i]*array2[(i % N)*N + int(i/N)]
    return new_array


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 4, 6, 8])
    N = 2
    print(a.reshape(N, N))

    print(trial(a, b, N).reshape(N, N))
