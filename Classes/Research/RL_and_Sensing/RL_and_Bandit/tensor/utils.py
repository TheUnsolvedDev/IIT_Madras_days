import numpy as np
import tensorflow as tf

from config import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class History:
    def __init__(self, map_size=ENV_SIZE, lambda_=LAMBDA):
        self.map_size = map_size
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        self.V = self.lambda_*np.eye(self.map_size*self.map_size)
        self.A, self.r = [], []

    def compute_uncertainity_reconstruction(self):
        if len(self.A) == 0:
            uncertainity = np.linalg.eigh(self.V)[0]
            reconstruction = np.ones((self.map_size, self.map_size))*1e-4
            uncertainity, reconstruction = np.array(
                uncertainity), np.array(reconstruction)
            return uncertainity.reshape(self.map_size, self.map_size), reconstruction.reshape(self.map_size, self.map_size)
        else:
            A, r = np.array(self.A), np.array(self.r)
            A, r = A.reshape(len(A), -1), r.reshape(len(r), -1)
            V = tf.matmul(A, A, transpose_a=True) + self.V
            uncertainity = tf.linalg.eigh(V)[1][0]
            reconstruction = tf.linalg.inv(V)@A.T@r
            uncertainity, reconstruction = np.array(
                uncertainity), np.array(reconstruction)
            rank = tf.linalg.matrix_rank(A)
            return uncertainity.reshape(self.map_size, self.map_size), reconstruction.reshape(self.map_size, self.map_size), rank

    def add(self, A_t, r_t):
        self.A.append(A_t)
        self.r.append(r_t)


class ReplayBuffer:
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }

    def __len__(self):
        return len(self.buffer['states'])

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer['states']) >= self.buffer_size:
            for key in self.buffer:
                self.buffer[key].pop(0)
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(next_state)
        self.buffer['dones'].append(done)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.buffer['states']), batch_size)
        states = np.array([self.buffer['states'][i]
                          for i in indices], dtype=np.float32)
        actions = np.array([self.buffer['actions'][i]
                           for i in indices], dtype=np.int32)
        rewards = np.array([self.buffer['rewards'][i]
                           for i in indices], dtype=np.float32)
        next_states = np.array([self.buffer['next_states'][i]
                               for i in indices], dtype=np.float32)
        dones = np.array([self.buffer['dones'][i]
                         for i in indices], dtype=np.float32)
        return states, actions, rewards, next_states, dones


def bresenham_map(transmitter, receiver, size):
    tx, ty = transmitter
    rx, ry = receiver
    map = np.zeros((size, size))
    dx = np.abs(rx - tx)
    dy = np.abs(ry - ty)
    sx = np.sign(rx - tx)
    sy = np.sign(ry - ty)
    err = dx - dy
    while True:
        map[tx, ty] = 1
        if tx == rx and ty == ry:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            tx += sx
        if e2 < dx:
            err += dx
            ty += sy
    return map


if __name__ == '__main__':
    map = bresenham_map((0, 0), (5, 8), 10)
    print(map)
