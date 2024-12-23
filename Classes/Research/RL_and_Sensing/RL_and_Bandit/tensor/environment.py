import numpy as np
import matplotlib.pyplot as plt
import pprint

from utils import *
from config import *


class Field:
    def __init__(self, size=ENV_SIZE, sparsity=SPARSITY, seed=32):
        self.size = size
        self.sparsity = sparsity
        self.seed = seed
        self.reset()

        self.valid_actions_placements = [(i, j) for i in range(
            self.size*self.size-1) for j in range(i+1, self.size*self.size)]
        self.valid_actions = [self.convert_action_to_placements(
            *action) for action in self.valid_actions_placements]
        self.actions_to_index = {action: i for i,
                                 action in enumerate(self.valid_actions)}
        self.index_to_action = {i: action for i,
                                action in enumerate(self.valid_actions)}
        self.num_actions = len(self.actions_to_index)

    def convert_action_to_placements(self, transmitter, receiver):
        return (transmitter//self.size, transmitter % self.size), (receiver//self.size, receiver % self.size)

    def convert_index_to_map(self, index):
        transmitter, receiver = self.index_to_action[index]
        return bresenham_map(transmitter, receiver, self.size)

    def reset(self):
        np.random.seed(self.seed)
        self.theta_star = np.random.uniform(0, 1, (self.size, self.size))
        self.theta_star = np.where(
            self.theta_star > self.sparsity, 1.0, 0.0).astype(np.float32)
        self.theta_star += np.random.normal(0, 0.03,
                                            (self.size, self.size)).astype(np.float32)
        self.theta_star = np.clip(self.theta_star, 0, 1).astype(np.float32)
        return self.theta_star

    def calculate_reward(self, step, action, rank, reconstruction, factors=[0.1, 1, 1, 1]):
        transmitter, receiver = self.index_to_action[action]
        manhattan_distance = np.abs(
            transmitter[0] - receiver[0]) + np.abs(transmitter[1] - receiver[1])
        mse = np.mean((self.theta_star - reconstruction)**2)

        facts = {
            'step': step,
            'manhattan_distance': float(manhattan_distance),
            'rank': float(rank),
            'mse': float(mse)
        }
        done = True if mse < 0.2 else False
        done |= True if step == MAX_EPISODE_STEPS else False
        
        factor1 = factors[0] * (-step)
        factor2 = factors[1] * (-manhattan_distance)
        factor3 = factors[2] * rank.numpy()
        factor4 = factors[3] * (-mse)
        # print(factor1, factor2, factor3, factor4)
        return float(factor1 + factor2 + factor3 + factor4), done, facts

    def step(self, action):
        if action < 0 or action >= len(self.valid_actions):
            raise Exception('Invalid action')
        map = self.convert_index_to_map(action)
        self.theta_star += np.random.normal(0, 0.01,
                                            (self.size, self.size)).astype(np.float32)
        reward = np.sum(map * self.theta_star)
        return map, reward


if __name__ == '__main__':
    field = Field()
    print(field.theta_star)
    pprint.pprint(field.valid_actions)
    pprint.pprint(field.actions_to_index)
    # plt.imshow(field.theta_star)
    # plt.show()
