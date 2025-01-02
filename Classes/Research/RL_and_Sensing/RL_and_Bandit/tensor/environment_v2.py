import numpy as np
import matplotlib.pyplot as plt
import pprint

from utils import *
from config import *


class Field:
    def __init__(self, size=ENV_SIZE, sparsity=SPARSITY, field_division=FIELD_DIVISION, seed=0):
        self.size = size
        self.sparsity = sparsity
        self.seed = seed
        self.field_division = field_division
        self.reset()

        if field_division <= 1 or field_division > size:
            raise ValueError(
                "Field division must be greater than 1 and less than or equal to the size.")
        self.action_sliced = np.array_split(np.arange(size), field_division)
        self.new_size = self.field_division

        self.valid_action_placement_indices = [(i, j) for i in range(
            self.new_size*self.new_size - 1) for j in range(i+1, self.new_size*self.new_size)]
        self.valid_action_indices = [self.convert_action_to_placements(
            *action) for action in self.valid_action_placement_indices]
        self.action_to_index_indices = {
            action: i for i, action in enumerate(self.valid_action_indices)}
        self.index_to_action_indices = {
            i: action for i, action in enumerate(self.valid_action_indices)}
        self.num_actions = len(self.action_to_index_indices)

    def convert_action_to_placements(self, transmitter, receiver):
        return (transmitter//self.new_size, transmitter % self.new_size), (receiver//self.new_size, receiver % self.new_size)

    def reset(self):
        # np.random.seed(self.seed)
        rng = np.random.RandomState(self.seed)
        self.theta_star = rng.uniform(0, 1, (self.size, self.size))
        self.theta_star = np.where(
            self.theta_star > self.sparsity, 1.0, 0.0).astype(np.float32)
        self.theta_star += np.random.normal(0, 0.02,
                                            (self.size, self.size)).astype(np.float32)
        self.theta_star = np.clip(self.theta_star, 0, 1).astype(np.float32)
        return self.theta_star

    def convert_index_to_map(self, index):
        if index < 0 or index >= len(self.valid_action_indices):
            raise Exception('Invalid action')
        transmitter, receiver = self.index_to_action_indices[index]
        new_transmitter_x = np.random.randint(self.action_sliced[transmitter[0]].min(
        ), self.action_sliced[transmitter[0]].max()+1)
        new_transmitter_y = np.random.randint(self.action_sliced[transmitter[1]].min(
        ), self.action_sliced[transmitter[1]].max()+1)
        new_receiver_x = np.random.randint(
            self.action_sliced[receiver[0]].min(), self.action_sliced[receiver[0]].max()+1)
        new_receiver_y = np.random.randint(
            self.action_sliced[receiver[1]].min(), self.action_sliced[receiver[1]].max()+1)
        new_transmitter = (new_transmitter_x, new_transmitter_y)
        new_receiver = (new_receiver_x, new_receiver_y)
        return bresenham_map(new_transmitter, new_receiver, self.size)

    def calculate_reward(self, step, action, rank, reconstruction, factors=[0.1, 1, 1, 1]):
        transmitter, receiver = self.index_to_action_indices[action]
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
        return float(factor1 + factor2 + factor3 + factor4), done, facts
    
    def step(self, action):
        if action < 0 or action >= self.num_actions:
            raise Exception('Invalid action')
        map = self.convert_index_to_map(action)
        theta_star = self.theta_star + np.random.normal(0, 0.01,
                                            (self.size, self.size)).astype(np.float32)
        reward = np.sum(map * theta_star)
        return map, reward


if __name__ == '__main__':
    field = Field(size=ENV_SIZE, sparsity=SPARSITY, field_division=FIELD_DIVISION)
    print(field.action_sliced)
    print(field.num_actions)
    # pprint.pprint(field.action_to_index_indices)
    print(field.convert_index_to_map(1))
    plt.imshow(field.theta_star,cmap='viridis')
    plt.show()
