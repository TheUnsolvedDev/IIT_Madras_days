import numpy as np
import tensorflow as tf
import os

from config import *
from model import *
from utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class QAgent:
    def __init__(self, img_size=ENV_SIZE, num_actions=10, learning_rate=ALPHA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        self.img_size = img_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size

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

    def update_target_model(self):
        self.target_Q_model.set_weights(self.Q_model.get_weights())

    def save_weights(self, name):
        os.makedirs('weights', exist_ok=True)
        self.Q_model.save_weights(f'weights/Q_model_{name}.weights.h5')

    def load_weights(self, name):
        self.Q_model.load_weights(f'weights/Q_model_{name}.weights.h5')


class DQNAgent(QAgent):
    def __init__(self, img_size=ENV_SIZE, num_actions=10, learning_rate=ALPHA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        super().__init__(img_size, num_actions, learning_rate, buffer_size, batch_size)
        self.Q_model = QModel(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.target_Q_model = QModel(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.update_target_model()

    @tf.function
    def td_update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.Q_model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            next_q_values = self.target_Q_model(next_states)
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + (1.0 - dones) * GAMMA * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        grads = tape.gradient(loss, self.Q_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Q_model.trainable_variables)
        )
        return loss


class DuelingDQNAgent(QAgent):
    def __init__(self, img_size=ENV_SIZE, num_actions=10, learning_rate=ALPHA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        super().__init__(img_size, num_actions, learning_rate, buffer_size, batch_size)
        self.Q_model = QModelAdvantage(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.target_Q_model = QModelAdvantage(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.update_target_model()

    @tf.function
    def td_update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.Q_model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            next_q_values = self.target_Q_model(next_states)
            next_q_values = tf.reduce_max(next_q_values, axis=1)
            next_q_values = tf.stop_gradient(next_q_values)
            target_q_values = rewards + (1.0 - dones) * GAMMA * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        grads = tape.gradient(loss, self.Q_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Q_model.trainable_variables)
        )
        return loss


class DoubleDQNAgent(QAgent):
    def __init__(self, img_size=ENV_SIZE, num_actions=10, learning_rate=ALPHA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        super().__init__(img_size, num_actions, learning_rate, buffer_size, batch_size)
        self.Q_model = QModel(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.target_Q_model = QModel(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.update_target_model()

    @tf.function
    def td_update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.Q_model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)

            next_actions = tf.argmax(self.Q_model(next_states), axis=1)
            next_q_values = self.target_Q_model(next_states)
            next_q_values = tf.reduce_sum(
                next_q_values * tf.one_hot(next_actions, self.num_actions), axis=1)
            next_q_values = tf.stop_gradient(next_q_values)

            target_q_values = rewards + (1.0 - dones) * GAMMA * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        grads = tape.gradient(loss, self.Q_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Q_model.trainable_variables))
        return loss


class DuelingDoubleDQNAgent(QAgent):
    def __init__(self, img_size=ENV_SIZE, num_actions=10, learning_rate=ALPHA, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
        super().__init__(img_size, num_actions, learning_rate, buffer_size, batch_size)
        self.Q_model = QModelAdvantage(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.target_Q_model = QModelAdvantage(
            inputs=(img_size, img_size, 2), outputs=num_actions)
        self.update_target_model()

    @tf.function
    def td_update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.Q_model(states)
            actions = tf.squeeze(actions)
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)

            next_actions = tf.argmax(self.Q_model(next_states), axis=1)
            next_q_values = self.target_Q_model(next_states)
            next_q_values = tf.reduce_sum(
                next_q_values * tf.one_hot(next_actions, self.num_actions), axis=1)
            next_q_values = tf.stop_gradient(next_q_values)

            target_q_values = rewards + (1.0 - dones) * GAMMA * next_q_values
            loss = tf.reduce_mean(tf.square(target_q_values - q_values))

        grads = tape.gradient(loss, self.Q_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.Q_model.trainable_variables))
        return loss


if __name__ == '__main__':
    agent = QAgent()
    agent.summary()

    for i in range(10):
        print(agent.update_epsilon(1, 0, i, 10, 0.1))
