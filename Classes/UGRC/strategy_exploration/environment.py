import jax
import numpy as np
import jax.numpy as jnp

from utils import *

import numpy as np
import random


def generate_building_room_sizes(num_buildings, map_size, min_rooms=2, max_rooms=5, min_size=2, max_size=10, seed=None):
    """
    Generates random room sizes for multiple buildings that fit within a square map.
    
    Parameters:
        num_buildings (int): The number of buildings to generate.
        map_size (int): The size of the square map (both width and height).
        min_rooms (int): The minimum number of rooms per building.
        max_rooms (int): The maximum number of rooms per building.
        min_size (int): The minimum dimension (width or height) of a room.
        max_size (int): The maximum dimension (width or height) of a room.
        seed (int, optional): A seed for the random number generator to ensure reproducibility.
    
    Returns:
        list of lists of tuples: A list where each element is a list of room dimensions 
        for a single building. Each tuple represents (width, height) of a room.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    buildings = []
    
    for _ in range(num_buildings):
        building_rooms = []
        num_rooms = random.randint(min_rooms, max_rooms)
        
        remaining_area = map_size ** 2
        for _ in range(num_rooms):
            max_dim = int(remaining_area ** 0.5)
            
            # Ensure valid range for width and height
            if min_size > max_dim:
                break
            
            width = random.randint(min_size, min(max_size, max_dim))
            height = random.randint(min_size, min(max_size, max_dim))
            
            # Ensure that width and height are valid and non-zero
            width = max(min_size, width)
            height = max(min_size, height)
            
            room_area = width * height
            
            if room_area <= remaining_area:
                building_rooms.append((width, height))
                remaining_area -= room_area
            else:
                # Reduce the room dimensions if the area left is too small
                if remaining_area > 0:
                    width = min(width, int(remaining_area ** 0.5))
                    height = min(height, remaining_area // width)
                    building_rooms.append((width, height))
                    remaining_area -= width * height
                break
        
        buildings.append(building_rooms)
    
    return buildings

def create_combined_building_layout(building_room_sizes, map_size, seed=None):
    """
    Generates a single binary matrix representing the top view of multiple buildings 
    within a square map.
    
    Parameters:
        building_room_sizes (list of lists of tuples): A list where each element is a list of room dimensions 
        for a single building. Each tuple represents (width, height) of a room.
        map_size (int): The size of the square map (both width and height).
        seed (int, optional): A seed for the random number generator to ensure reproducibility.
    
    Returns:
        np.array: A binary matrix representing the top view of all buildings on a single map.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    combined_matrix = np.zeros((map_size, map_size), dtype=int)
    
    for rooms in building_room_sizes:
        placed = False
        for _ in range(100):  # Try 100 times to place the building in a random position
            start_y = random.randint(0, map_size - 1)
            start_x = random.randint(0, map_size - 1)
            
            if all(start_y + height <= map_size and start_x + width <= map_size
                   for width, height in rooms):
                temp_matrix = np.zeros((map_size, map_size), dtype=int)
                
                current_y = start_y
                for width, height in rooms:
                    if np.any(temp_matrix[current_y:current_y + height, start_x:start_x + width] == 1):
                        break
                    temp_matrix[current_y:current_y + height, start_x:start_x + width] = 1
                    current_y += height
                
                if np.all(combined_matrix + temp_matrix <= 1):  # Check for overlaps
                    combined_matrix += temp_matrix
                    placed = True
                    break
    
    return combined_matrix

class Environment:
    def __init__(self, sparsity=0.2, size=10, rng=1) -> None:
        self.sparsity = sparsity
        self.size = size
        self.rng = rng
        self.reset()

    def reset(self):
        num_buildings = 5
        building_room_sizes = generate_building_room_sizes(
            num_buildings, self.size, seed=self.rng)
        self.map_star = create_combined_building_layout(
            building_room_sizes, self.size, seed=self.rng)

        self.sparsity = np.mean(self.map_star.flatten())

        def n_choose_2(n):
            if n < 2:
                return 0  # If n < 2, there are no ways to choose 2 items
            return (n * (n - 1)) // 2
        self.total_actions = self.size**4

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
