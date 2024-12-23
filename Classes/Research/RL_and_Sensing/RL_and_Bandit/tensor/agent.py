import numpy as np
import tensorflow as tf

from config import *
from model import *


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
        states = np.array([self.buffer['states'][i] for i in indices])
        actions = np.array([self.buffer['actions'][i] for i in indices])
        rewards = np.array([self.buffer['rewards'][i] for i in indices])
        next_states = np.array([self.buffer['next_states'][i]
                               for i in indices])
        dones = np.array([self.buffer['dones'][i] for i in indices])
        return states, actions, rewards, next_states, dones


class QAgent:
    def __init__(self, img_size=ENV_SIZE, num_actions=10, learning_rate=ALPHA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.img_size = img_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.Q_model = QModel(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.target_Q_model = QModel(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.update_target_model()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=self.buffer_size)

    def summary(self):
        self.Q_model.summary()
        self.target_Q_model.summary()

    @tf.function
    def model_action(self, state):
        state = tf.expand_dims(state, axis=0)
        return self.Q_model(state)

    def random_action(self, state):
        return np.random.choice(self.num_actions)

    def epsilon_greedy_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.random_action(state)
        else:
            return np.argmax(self.model_action(state))

    # @tf.function
    def td_update(self, states, actions, rewards, next_states, dones):
        # actions = tf.cast(actions, tf.float32)
        with tf.GradientTape() as tape:
            q_values = self.Q_model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            next_q_values = self.target_Q_model(next_states)
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        grads = tape.gradient(loss, self.Q_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Q_model.trainable_variables)
        )
        return loss.numpy()

    def update_target_model(self):
        self.target_Q_model.set_weights(self.Q_model.get_weights())


if __name__ == '__main__':
    agent = QAgent()
    agent.summary()

    for i in range(10):
        print(agent.update_epsilon(1, 0, i, 10, 0.1))
