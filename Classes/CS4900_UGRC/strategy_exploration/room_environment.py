import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import Tuple

from utils import *


class CreateRooms:
    def __init__(self,
                 type: int = 1,  # type: ignore
                 size: int = 13) -> None:
        """
        Args:
            type (int): A random seed for the room creation
            size (int): The size of the room
        """
        assert size >= 10, "The size of the room must be at least 10"
        self.size = size
        self.type = type
        self.wall_attenuation = 5
        self.objects_attenuation = 2
        self.prng = jax.random.PRNGKey(self.type)
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets the room to its initial state and returns the current state.

        Returns:
            np.ndarray: The current state of the room.
        """
        self.room = np.zeros((self.size, self.size), dtype=np.float32)
        self.room[:, 0] = self.room[:, -1] = self.room[0,
                                                       :] = self.room[-1, :] = self.wall_attenuation
        if self.size % 2 == 0:
            mids = [np.floor((self.size-2) / 2).astype(int),
                    np.ceil((self.size) / 2).astype(int)]
        else:
            mids = [np.floor((self.size) / 2).astype(int),
                    np.ceil((self.size-2) / 2).astype(int)]
        self.room[:, mids[0]] = self.room[:, mids[1]
                                          ] = self.room[mids[0], :] = self.room[mids[1], :] = self.wall_attenuation

        num_doors_per_quad = 3
        prng = self.prng

        for door in range(num_doors_per_quad):
            prng = jax.random.split(prng, 12)
            make_doors_quad1 = [jax.random.choice(prng[0], np.arange(
                1, mids[0]-1)), jax.random.choice(prng[1], np.arange(1, mids[0]-1))]
            make_doors_quad2 = [jax.random.choice(prng[2], np.arange(
                1, mids[0]-1)), jax.random.choice(prng[3], np.arange(mids[0]+1, self.size-1))]
            make_doors_quad3 = [jax.random.choice(prng[4], np.arange(
                mids[0]+1, self.size-1)), jax.random.choice(prng[5], np.arange(1, mids[0]-1))]
            make_doors_quad4 = [jax.random.choice(prng[6], np.arange(
                mids[0]+1, self.size-1)), jax.random.choice(prng[7], np.arange(mids[0]+1, self.size))]
            choose_side = jax.random.randint(
                prng[8], minval=0, maxval=2, shape=(1,))[0]
            self.room[choose_side*make_doors_quad1[0],
                      (1-choose_side)*make_doors_quad1[1]] = 0
            choose_side = jax.random.randint(
                prng[9], minval=0, maxval=2, shape=(1,))[0]
            self.room[choose_side*make_doors_quad2[0],
                      (1-choose_side)*make_doors_quad2[1]] = 0
            choose_side = jax.random.randint(
                prng[10], minval=0, maxval=2, shape=(1,))[0]
            self.room[choose_side*make_doors_quad3[0],
                      (1-choose_side)*make_doors_quad3[1]] = 0
            choose_side = jax.random.randint(
                prng[11], minval=0, maxval=2, shape=(1,))[0]
            self.room[choose_side*make_doors_quad4[0],
                      (1-choose_side)*make_doors_quad4[1]] = 0
            prng = prng[-1]

        prng = jax.random.split(self.prng, num=2)
        fill_quads = jax.random.randint(
            prng[1], minval=0, maxval=2, shape=(4,))
        num_objects_per_quad = 5
        prng = prng[-1]

        for quad, fill in enumerate(fill_quads):
            for objects in range(num_objects_per_quad):
                prng = jax.random.split(prng, 8)
                if fill != 0 and quad == 0:
                    make_objects_quad = [jax.random.choice(prng[0], np.arange(
                        1, mids[0]-1)), jax.random.choice(prng[1], np.arange(1, mids[0]-1))]
                    self.room[make_objects_quad[0], make_objects_quad[1]
                              ] = self.objects_attenuation
                elif fill != 0 and quad == 1:
                    make_objects_quad = [jax.random.choice(prng[2], np.arange(
                        1, mids[0]-1)), jax.random.choice(prng[3], np.arange(mids[0]+1, self.size-1))]
                    self.room[make_objects_quad[0], make_objects_quad[1]
                              ] = self.objects_attenuation
                elif fill != 0 and quad == 2:
                    make_objects_quad = [jax.random.choice(prng[4], np.arange(
                        mids[0]+1, self.size-1)), jax.random.choice(prng[5], np.arange(1, mids[0]-1))]
                    self.room[make_objects_quad[0], make_objects_quad[1]
                              ] = self.objects_attenuation
                elif fill != 0 and quad == 3:
                    make_objects_quad = [jax.random.choice(prng[6], np.arange(
                        mids[0]+1, self.size-1)), jax.random.choice(prng[7], np.arange(mids[0]+1, self.size))]
                    self.room[make_objects_quad[0], make_objects_quad[1]
                              ] = self.objects_attenuation
                prng = prng[-1]

        self.sparsity = self.room.astype(bool).astype(int).mean()
        self.room += np.random.normal(0, 0.1, size=self.room.shape)
        self.total_actions = self.size**4
        self.room = np.maximum(self.room, 0)
        self.map_star = self.room
        self.map_star = (self.map_star - self.map_star.min()) / \
            (self.map_star.max() - self.map_star.min())
        # self.plot_room(self.map_star)
        return self.map_star

    def _action_vector(self, number: int) -> np.ndarray:
        """
        Returns a boolean vector representing the action taken.

        Args:
            number: Integer representing the action taken.

        Returns:
            np.ndarray: Boolean vector representing the action taken.
        """
        action = convert_actions(number, self.size)
        cut_indices = bresenham_line(*action)
        mask = np.zeros_like(self.map_star)
        rows, cols = zip(*cut_indices)
        mask[rows, cols] = 1
        return mask.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float]:
        """
        Perform the action on the environment and return the resulting mask and reward.

        Args:
            action (int): The action taken by the agent.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing the mask of the cuts made and the reward received.
        """
        if 0 < action and action >= self.total_actions:
            raise ValueError(
                f"Not in the range of total actions [{0}:{self.total_actions})")
        mask = self._action_vector(action)
        reward = self._reward_value(mask)
        return mask, reward


    def _reward_value(self, action_vector: np.ndarray) -> float:
        """
        Computes the reward value for the given action.

        Args:
            action_vector (np.ndarray): The boolean vector representing the action taken.

        Returns:
            float: The reward value.
        """
        reward = np.dot(self.map_star.flatten(), action_vector) + np.random.normal(0, 0.01)
        return reward


    def plot_room(self, array: np.ndarray) -> None:
        """
        Plots the room with the given array.

        Args:
            array (np.ndarray): The array to plot.

        Returns:
            None
        """
        plt.imshow(array)
        plt.show()


if __name__ == '__main__':
    images = []
    for i in range(9):
        rooms = CreateRooms(type=i,size = 20)
        images.append(rooms.map_star)
        print(rooms._action_vector(10700).reshape(rooms.size, rooms.size))
        
    plt.imshow(rooms._action_vector(10708).reshape(rooms.size, rooms.size))
    # plt.tight_layout()
    plt.title('Action 10708 to bresenham line')
    # plt.axes('off')
    plt.savefig('plots/action.png')
    plt.show()
    
    # fig,ax = plt.subplots(3,3,figsize=(10,10))
    # plt.tight_layout()
    # for i in range(3):
    #     for j in range(3):
    #         ax[i,j].imshow(images[i*3+j])
    #         ax[i,j].axis('off')
    #         ax[i,j].set_title('Room {}'.format(i*3+j+1))
    # plt.savefig('plots/rooms.png')
    # plt.show()
    
