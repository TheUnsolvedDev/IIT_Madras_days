import jax
import jax.numpy as jnp
import flax
import numpy as np

from utils import bresenham_line


class Field:
    def __init__(self, rng, size=3):
        self.size = size
        self.rng = rng
        self.reset()
        self.valid_actions_placements = [(i, j) for i in range(
            self.size*self.size-1) for j in range(i+1, self.size*self.size)]
        self.valid_actions = [((i//self.size, i % self.size), (j//self.size, j % self.size))
                              for i, j in self.valid_actions_placements]
        self.action_to_index = {}
        self.index_to_action = {}
        for i in range(len(self.valid_actions)):
            self.action_to_index[self.valid_actions[i]] = i
            self.index_to_action[i] = self.valid_actions[i]
        self.num_total_actions = len(self.valid_actions)
        

    def convert_action_index_to_map(self, index):
        tx, rx = self.index_to_action[index]
        return bresenham_line(tx, rx, map_size=self.size)

    def reset(self):
        self.theta_star = jax.random.bernoulli(
            self.rng, 0.5, shape=(self.size, self.size))
        self.theta_star = self.theta_star.astype(jnp.float32)
        self.theta_star += 0.01*jax.random.normal(
            self.rng, shape=(self.size, self.size))
        self.theta_star = jnp.clip(self.theta_star, 0, 1)
        return self.theta_star

    def step(self, action):
        if action < 0 or action >= self.num_total_actions:
            raise ValueError(f"Action {action} is not valid.")
        tx, rx = self.index_to_action[action]
        maps = bresenham_line(tx, rx, map_size=self.size)
        result = self.theta_star * maps
        return jnp.sum(result)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    field = Field(rng)
    print(field.reset())
    for i in range(10):
        action = np.random.choice(np.arange(field.num_total_actions))
        print(action)
        print(f"Action: {action}, Reward: {field.step(action)}")
