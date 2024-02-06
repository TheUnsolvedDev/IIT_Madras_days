import jax
import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

key = jax.random.PRNGKey(0)


def generate_map(n, sparsity, seed=0):
    np.random.seed(seed)
    num_zeros = int(n**2 * sparsity)
    binary_map = np.ones((n, n))
    indices_to_set_zero = np.random.choice(n**2, num_zeros, replace=False)
    binary_map.flat[indices_to_set_zero] = 0
    return binary_map


def LOG_kernel(size=5):
    size = [size+2, size+2]
    # log_kernel = np.array([
    #     [-1, -1, -1],
    #     [-1, 8, -1],
    #     [-1, -1, -1]
    # ])
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

    return np.array(L)


def convert_actions(num, to_base):
    temp = np.zeros(4, dtype=np.int16)
    count = 0
    while num > 0:
        digit = num % to_base
        num //= to_base
        temp[count] = digit
        count += 1
    return temp[::-1]


def bresenham_line(x1, y1, x2, y2):
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

    return line_points


def write_value(value):
    with open('maps.pkl', 'wb') as f:
        pickle.dump(value, f)


def read_value():
    with open('maps.pkl', 'rb') as f:
        val = pickle.load(f)
    return val


def calculate_ssim(img1, img2):
    img1 = np.clip(np.array(img1)*255, 0, 255).astype(np.uint8)
    img2 = np.clip(np.array(img2)*255, 0, 255).astype(np.uint8)
    return ssim(img1, img2)


def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


if __name__ == '__main__':
    for i in LOG_kernel(8):
        print(i.reshape(8, 8))
