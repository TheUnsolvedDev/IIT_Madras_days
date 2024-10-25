import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from environment import Env1D
from solver import Solver
from utils import *

class Algorithm:
    def __init__(self, max_iteration=1000) -> None:
        self.max_iteration = max_iteration
        self.env = Env1D()
        self.solver = Solver(max_iteration)
        self.all_mask_coord = [
            (x, y) for x in range(self.env.size) for y in range(self.env.size)]
        self.all_mask = jnp.array([
            bresenham1D(x, y, self.env.size) for x, y in self.all_mask_coord])
        
    def init(self):
        self.theta_star = self.env.reset()
        self.A,self.b = [],[]
        
    
    def simulate(self):
        theta = jnp.zeros_like(self.theta_star)
        action = 10
        for i in range(self.max_iteration):
            mask,result = self.env.act(action)
            self.A.append(mask)
            self.b.append(result)
            theta,prod = self.solver.solve_tikhonov(self.A, self.b, 0.01)
            theta += jax.random.normal(self.env.rng, (self.env.size,))*0.01
            theta = jnp.clip(theta, 0, 1)
            
            if i % 1 == 0:
                print('action taken:',idx_to_coord(action, self.env.size)[0],idx_to_coord(action, self.env.size)[1])
                hmap = np.vstack([mask,theta, self.theta_star])
                plt.imshow(hmap)
                plt.show()
                
            eigen_values,eigen_vectors = jnp.linalg.eigh(prod)
            action = jnp.argmin(self.all_mask@eigen_vectors[0])

if __name__ == "__main__":
    algorithm = Algorithm()
    algorithm.init()
    algorithm.simulate()