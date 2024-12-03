import numpy as np

def bresenham1D(x0, x1, size):
    mask = np.zeros(size)
    if x0 > x1:
        x0, x1 = x1, x0
    mask[x0:x1] = 1
    return mask

def bresenham2D(x1: int, y1: int, x2: int, y2: int, size: int):
    line_points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = dx - dy

    while True:
        line_points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        double_error = 2 * error
        if double_error > -dy:
            error -= dy
            x1 += step_x
        if double_error < dx:
            error += dx
            y1 += step_y
    mask = np.zeros((size, size))
    for x, y in line_points:
        mask[x, y] = 1
    return mask.reshape(-1,)

def coord_to_idx_1D(x, y, size):
    return y * size + x


def idx_to_coord_1D(idx, size):
    y = idx // size
    x = idx - y * size
    return x, y

def idx_to_coord_2D(num: int, to_base: int) -> np.ndarray:
    temp = np.zeros(4, dtype=np.int16)
    count = 0
    while num > 0:
        digit = num % to_base
        num //= to_base
        temp[count] = digit
        count += 1
    return temp[::-1]



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