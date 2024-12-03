import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from environment import Env1D
from solver import Solver
from utils import *


class Algorithm:
    def __init__(self, max_iteration=100) -> None:
        self.max_iteration = max_iteration
        self.env = Env1D(size=30)
        self.solver = Solver(max_iteration)
        self.all_mask_coord = [
            (x, y) for x in range(self.env.size) for y in range(self.env.size)]
        self.all_mask = jnp.array([
            bresenham1D(x, y, self.env.size) for x, y in self.all_mask_coord])

    def init(self):
        self.theta_star = self.env.reset()
        self.A, self.b = [], []

    def simulate(self, kernel='sharpen'):
        theta = jnp.zeros_like(self.theta_star)
        rank_history = []
        action = 10
        for i in range(self.max_iteration+1):
            mask, result = self.env.act(action)
            self.A.append(mask)
            self.b.append(result)
            if kernel == 'sharpen':
                theta, prod = self.solver.solve_Sharpen_kernel(
                    self.A, self.b, 0.01)
            elif kernel == 'log':
                theta, prod = self.solver.solve_LOG_kernel(
                    self.A, self.b, 0.01)
            elif kernel == 'tikhonov':
                theta, prod = self.solver.solve_tikhonov(self.A, self.b, 0.01)
            theta += jax.random.normal(self.env.rng, (self.env.size,))*(1e-4)
            theta = jnp.clip(theta, 0, 1)

            rank = jnp.linalg.matrix_rank(jnp.array(self.A))
            rank_history.append(rank)
            
            if i % 20 == 0:
                print('steps:', i, '\taction taken:', idx_to_coord(action, self.env.size)[
                      0], idx_to_coord(action, self.env.size)[1], '\trank:', rank)
                hmap = np.vstack([mask, self.theta_star, theta])
                plt.imshow(hmap)
                plt.show()

            eigen_values, eigen_vectors = jnp.linalg.eigh(prod)
            action = jnp.argmin(self.all_mask@eigen_vectors[0])
        return rank_history


if __name__ == "__main__":
    algorithm = Algorithm()
    algorithm.init()
    algorithm.simulate()
