import jax
import numpy as np
import jax.numpy as jnp

from utils import *


class Environment:
    def __init__(self, type, sparsity=0.2, size=10) -> None:
        self.sparsity = sparsity
        self.size = size
        self.type = type
        self.reset()

    def reset(self):
        if self.type == 0:
            self.map_star = np.zeros((self.size, self.size))
            self.map_star[self.size//2 - self.size//4:self.size//2 + self.size //
                          4, self.size//2 - self.size//4:self.size//2 + self.size//4] = 1
            self.sparsity = np.mean(self.map_star.flatten())
        elif self.type == 1:
            self.map_star = np.zeros((self.size, self.size))
            center = self.size // 2
            if self.size % 2 == 0:
                half_width = self.size // 4
                half_length = self.size // 4
            else:
                half_width = (self.size // 4) - 1
                half_length = (self.size // 4) - 1
            self.map_star[center - half_length:center + half_length, :] = 1
            self.map_star[:, center - half_width:center + half_width] = 1
            self.sparsity = np.mean(self.map_star.flatten())
        else:
            self.map_star = generate_map(
                n=self.size, sparsity=self.sparsity, seed=self.type)
        self.total_actions = np.power(self.size, 4)

    def _action_vector(self, number):
        action = convert_actions(number, self.size)
        cut_indices = bresenham_line(*action)
        mask = np.zeros_like(self.map_star)
        rows, cols = zip(*cut_indices)
        mask[rows, cols] = 1
        return mask.flatten()

    def _reward_value(self, action_vector):
        # + np.random.normal(0, 0.01)
        reward = self.map_star.flatten()@action_vector + np.random.normal(0, 0.01)
        return reward

    def step(self, action):
        if 0 < action and action >= self.total_actions:
            raise ValueError(
                f"Not in the range of total actions [{0}:{self.total_actions})")
        mask = self._action_vector(action)
        reward = self._reward_value(mask)
        return mask, reward

    def rollout(self, action_list):
        actions, rewards = [], []
        for i in action_list:
            action, reward = self.step(i)
            actions.append(action)
            rewards.append(reward)
        return np.array(actions), np.array(rewards).reshape(-1, 1)


if __name__ == '__main__':
    env = Environment()
    print(env.map_star)
