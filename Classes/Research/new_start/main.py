import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from environment import *
from solver import *
from utils import *
import tqdm


class Algorithm:
    def __init__(self, env_type='1D', max_iteration=1000) -> None:
        self.max_iteration = max_iteration
        self.env_type = env_type
        if env_type == '1D':
            self.env = Env1D(size=30)
            self.all_mask_coord = [
                (x, y) for x in range(self.env.size) for y in range(self.env.size)]
            self.all_mask = jnp.array([
                bresenham1D(x, y, self.env.size) for x, y in self.all_mask_coord])
        elif env_type == '2D':
            self.env = Env2D(size=10)
            self.all_mask_coord = [(w, x, y, z) for w in range(self.env.map_size) for x in range(
                self.env.map_size) for y in range(self.env.map_size) for z in range(self.env.map_size) if (w,x) <= (y,z)]
            self.all_mask = jnp.array([
                bresenham2D(w, x, y, z, self.env.map_size) for w, x, y, z in tqdm.tqdm(self.all_mask_coord)])
        self.solver = Solver(max_iteration)

    def init(self):
        self.theta_star = self.env.reset()
        self.A, self.b = [], []

    def simulate(self, kernel='tikhonov'):
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

            if i % 50 == 0:
                if self.env_type == '1D':
                    print('steps:', i, '\trank:', rank)
                    hmap = np.vstack([mask, self.theta_star, theta])
                    plt.imshow(hmap)
                    plt.show()
                if self.env_type == '2D':
                    print('steps:', i, '\trank:', rank)
                    hmap = np.hstack([mask.reshape(self.env.map_size, self.env.map_size), self.theta_star.reshape(self.env.map_size, self.env.map_size), theta.reshape(self.env.map_size, self.env.map_size)])
                    plt.imshow(hmap)
                    plt.show()
            eigen_values, eigen_vectors = jnp.linalg.eigh(prod)
            action = jnp.argmin(self.all_mask@eigen_vectors[0])
        return rank_history


if __name__ == "__main__":
    algorithm = Algorithm(env_type='2D')
    algorithm.init()
    algorithm.simulate()
