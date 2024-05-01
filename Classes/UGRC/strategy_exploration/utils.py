import jax
import numpy as np
import jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
import skimage.metrics as skmetrics
from typing import *

key = jax.random.PRNGKey(0)


def generate_map(n: int, sparsity: float, seed: int = 0) -> np.ndarray:
    """
    Generate a binary map of size (n, n) with the given sparsity.

    Args:
        n (int): The size of the map.
        sparsity (float): The sparsity of the map.
        seed (int, optional): The random seed. Defaults to 0.

    Returns:
        np.ndarray: The generated binary map.
    """
    np.random.seed(seed)
    num_zeros = int(n**2 * sparsity)
    binary_map = np.ones((n, n), dtype=np.float32)
    indices_to_set_zero = np.random.choice(n**2, num_zeros, replace=False)
    binary_map.flat[indices_to_set_zero] = 0
    return binary_map


def LOG_kernel(size: int = 5) -> np.ndarray:
    """
    Generate a 2D LOG kernel of size (size, size) without border padding.

    Args:
        size (int): The size of the kernel.

    Returns:
        np.ndarray: The generated LOG kernel with shape (size*size, ).
    """
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

    return np.array(L)


def Gauss_kernel(size: int = 5) -> np.ndarray:
    """
    Generate a 2D Gauss kernel of size (size, size) without border padding.

    Args:
        size (int): The size of the kernel.

    Returns:
        np.ndarray: The generated LOG kernel with shape (size*size, ).
    """
    size = [size+2, size+2]
    log_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
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

def Sharpen_kernel(size: int = 5) -> np.ndarray:
    """
    Generate a 2D Sharpen kernel of size (size, size) without border padding.

    Args:
        size (int): The size of the kernel.

    Returns:
        np.ndarray: The generated LOG kernel with shape (size*size, ).
    """
    size = [size+2, size+2]
    log_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
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


def convert_actions(num: int, to_base: int) -> np.ndarray:
    """
    Converts an integer action number to a base-size representation.

    Args:
        num (int): The action number to be converted.
        to_base (int): The base to convert the action number to.

    Returns:
        np.ndarray: The converted action number with shape (4, ).
    """
    temp = np.zeros(4, dtype=np.int16)
    count = 0
    while num > 0:
        digit = num % to_base
        num //= to_base
        temp[count] = digit
        count += 1
    return temp[::-1]


def bresenham_line(x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
    """
    Returns a list of points representing a line between (x1, y1) and (x2, y2).

    Args:
        x1 (int): The x coordinate of the starting point.
        y1 (int): The y coordinate of the starting point.
        x2 (int): The x coordinate of the ending point.
        y2 (int): The y coordinate of the ending point.

    Returns:
        List[Tuple[int, int]]: A list of points representing the line.
    """
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


def write_value(value: Any) -> None:
    """
    Writes the given value to a file using pickle.

    Args:
        value: The value to be written to the file.

    Returns:
        None
    """
    with open('maps.pkl', 'wb') as f:
        pickle.dump(value, f)


def read_value() -> Any:
    """
    Reads a value from a file using pickle.

    Returns:
        Any: The deserialized value from the file.
    """
    with open('maps.pkl', 'rb') as f:
        val = pickle.load(f)
    return val


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculates the structural similarity between two images.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.

    Returns:
        float: The structural similarity index.
    """
    img1 = np.array(img1)*255.0
    img2 = np.array(img2)*255.0
    img1 = np.clip(np.array(img1), 0, 255)
    img2 = np.clip(np.array(img2), 0, 255)
    return skmetrics.structural_similarity(
        im1=img1, im2=img2, data_range=255, channel_axis=1)


def calculate_mse(imageA, imageB):
    return skmetrics.mean_squared_error(image0=imageA, image1=imageB)


def calculate_psnr(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Calculates the peak signal-to-noise ratio between two images.

    Args:
        image_true (np.ndarray): The ground truth image.
        image_test (np.ndarray): The image to be evaluated.

    Returns:
        float: The peak signal-to-noise ratio value.
    """
    image_true = np.array(image_true)*255.0
    image_test = np.array(image_test)*255.0
    image_true = np.clip(np.array(image_true), 0.1, 255)
    image_test = np.clip(np.array(image_test), 0.1, 255)
    return skmetrics.peak_signal_noise_ratio(image_true=image_true, image_test=image_test,
                                             data_range=255)


class DataLogger:
    def __init__(self, file_path: str) -> None:
        """
        Initializes a DataLogger.

        Args:
            file_path (str): The path to the log file.

        Returns:
            None
        """
        self.file_path = file_path
        self.count = 0
        with open(self.file_path, 'w') as f:
            f.write('step,action,psnr_value,mse_value,ssim_value,rank\n')

    def append_log(self, action: int, psnr_value: float, mse_value: float, ssim_value: float, rank: int) -> None:
        """
        Appends the given data to the log file.

        Args:
            action (int): The action taken.
            psnr_value (float): The PSNR value.
            mse_value (float): The MSE value.
            ssim_value (float): The SSIM value.
            rank (int): The rank of the action.

        Returns:
            None
        """
        self.count += 1
        with open(self.file_path, 'a') as file:
            file.write(
                f'{self.count},{action},{psnr_value},{mse_value},{ssim_value},{rank}\n')


if __name__ == '__main__':
    for i in LOG_kernel(3):
        print(i)
