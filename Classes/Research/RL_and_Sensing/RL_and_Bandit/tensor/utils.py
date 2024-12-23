import numpy as np

from config import *


def bresenham_map(transmitter, receiver, size):
    tx, ty = transmitter
    rx, ry = receiver
    map = np.zeros((size, size))
    dx = np.abs(rx - tx)
    dy = np.abs(ry - ty)
    sx = np.sign(rx - tx)
    sy = np.sign(ry - ty)
    err = dx - dy
    while True:
        map[tx, ty] = 1
        if tx == rx and ty == ry:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            tx += sx
        if e2 < dx:
            err += dx
            ty += sy
    return map

if __name__ == '__main__':
    map = bresenham_map((0,0),(5,8),10)
    print(map)