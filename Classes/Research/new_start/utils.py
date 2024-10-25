import numpy as np

def bresenham1D(x0, x1, size):
    mask = np.zeros(size)
    if x0 > x1:
        x0, x1 = x1, x0
    mask[x0:x1] = 1
    return mask

def coord_to_idx(x, y, size):
    return y * size + x


def idx_to_coord(idx, size):
    y = idx // size
    x = idx - y * size
    return x, y

def LOG_kernel(size: int = 5) -> np.ndarray:
    size = [size+2, size+2]
    log_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    row_traverse = size[0] - log_kernel.shape[0] + 1
    col_traverse = size[1] - log_kernel.shape[1] + 1

    L = []

    for row in range(row_traverse):
        for col in range(col_traverse):
            kernel = np.zeros(size)
            kernel[row:row+log_kernel.shape[0],
                   col:col+log_kernel.shape[1]] = log_kernel
            kernel = kernel[1:kernel.shape[0]-1, 1:kernel.shape[1]-1]
            L.append(kernel.flatten())

    return np.array(L)[:size[0]-2, :size[1]-2]

if __name__ == "__main__":
    print(LOG_kernel(3))