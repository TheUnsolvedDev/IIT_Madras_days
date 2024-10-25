import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import *


class Env1D:
    def __init__(self, size=30, seed=0):
        self.rng = jax.random.PRNGKey(seed)
        if size < 8:
            raise ValueError("size must be greater than 8")
        self.size = size
        self.max_action_ind = (self.size - 1) * self.size + (self.size-1)
        self.reset()

    def reset(self):
        self.map = jax.random.bernoulli(self.rng, 0.5, (self.size,))
        self.map = self.map.astype(jnp.int8)
        return self.map

    def act(self, action):
        int_to_location = idx_to_coord(action, self.size)
        mask = bresenham1D(int_to_location[0], int_to_location[1], self.size)
        return mask,jnp.sum(self.map * mask)


def main():
    env = Env1D()
    theta_star = env.reset()
    for i in range(100):
        action = np.random.randint(0, env.max_action_ind)
        mask,reward = env.act(action)
        print(action,mask,reward)
        


if __name__ == "__main__":
    main()
