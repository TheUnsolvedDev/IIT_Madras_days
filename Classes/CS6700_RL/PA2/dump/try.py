import jax
import numpy as np
import jax.numpy as jnp
import gymnasium as gym
import flax
from flax.training.train_state import TrainState
import optax
import functools
import matplotlib.pyplot as plt
import tqdm
import gc
import argparse

ALPHA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64
CAPACITY = 20000
UPDATE_EVERY = 20


def smooth_rewards(rewards, window_size=10):
    smoothed_rewards = np.zeros_like(rewards)
    for i in range(len(rewards)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(rewards), i + window_size // 2 + 1)
        smoothed_rewards[i] = np.mean(rewards[window_start:window_end])
    return smoothed_rewards


def plot_data(mean, std, name):
    x = range(len(mean))

    plt.plot(x, mean, color='blue', label='Mean')
    plt.plot(x, smooth_rewards(mean), color='orange', label='smoothed')
    plt.fill_between(x, mean - std, mean + std, color='blue',
                     alpha=0.3, label='Mean ± Std')

    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.savefig(name.replace(' ', '_')+'.png')
    # plt.show(block = False)
    plt.close()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


class ValueNetworkMean(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        value_stream = flax.linen.Dense(1)(x)
        advantage_stream = flax.linen.Dense(self.action_dim)(x)
        q_values = value_stream + \
            (advantage_stream - jnp.mean(advantage_stream, axis=-1, keepdims=True))

        return q_values


class ValueNetworkMax(flax.linen.Module):
    action_dim: int

    @flax.linen.compact
    def __call__(self, x: jnp.ndarray):
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        x = flax.linen.Dense(16)(x)
        x = flax.linen.gelu(x)
        value_stream = flax.linen.Dense(1)(x)
        advantage_stream = flax.linen.Dense(self.action_dim)(x)
        q_values = value_stream + \
            (advantage_stream - jnp.max(advantage_stream, axis=-1, keepdims=True))

        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(len(self.buffer), size=(batch_size,))

    def get_batch(self, indices):
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices])
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)
        return states, actions, rewards, next_states, dones


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class DuelingDQNMean:
    def __init__(self, env, num_actions, observation_shape, seed=0) -> None:
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.env = env

        self.value = ValueNetworkMean(num_actions)
        self.value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=self.value.init(self.rng, jnp.ones(observation_shape)),
            target_params=self.value.init(
                self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA)
        )
        self.value.apply = jax.jit(self.value.apply)
        self.value_state = self.value_state.replace(target_params=optax.incremental_update(
            self.value_state.params, self.value_state.target_params, 0.9))
        self.replay_buffer = ReplayBuffer(CAPACITY)
        self.counter = 1
        self.updates = 1

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_greedy(self, q_state, state):
        q_values = self.value.apply(q_state.params, state)
        action = q_values.argmax(axis=-1)[0]
        return action

    def sample(self, state, epsilon=0.1):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.num_actions)
        action = np.array(self.policy_greedy(self.value_state, state))
        return action

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, value_state, states, actions, rewards, next_states,  dones):
        value_next_target = self.value.apply(
            value_state.target_params, next_states)
        value_next_target = jnp.max(value_next_target, axis=-1)
        next_q_value = (rewards + (1 - dones) * GAMMA * value_next_target)

        def mse_loss(params):
            value_pred = self.value.apply(params, states)
            value_pred = value_pred[jnp.arange(
                value_pred.shape[0]), actions.squeeze()]
            return ((jax.lax.stop_gradient(next_q_value) - value_pred) ** 2).mean()

        loss_value, grads = jax.value_and_grad(
            mse_loss)(value_state.params)
        value_state = value_state.apply_gradients(grads=grads)
        return loss_value, value_state

    def train_single_step(self):
        state = self.env.reset(seed=self.seed)[0]
        key = self.rng
        epsilon = linear_schedule(
            start_e=1, end_e=0.05, duration=500, t=0 if self.counter <= 50 else self.counter-50)
        episode_loss, episode_rewards = 0, 0
        for _ in range(500):
            action = self.sample(np.expand_dims(state, axis=0), epsilon)
            next_state, reward, done, truncated, info = self.env.step(action)
            self.replay_buffer.push(
                [state, action, reward, next_state, done or truncated])
            state = next_state
            episode_rewards += reward

            if truncated or done:
                break

            if len(self.replay_buffer.buffer) > 128:
                indices = self.replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(
                    indices)
                loss_values, self.value_state = self.update(self.value_state,
                                                            states, actions, rewards, next_states,  dones)
                if self.updates % UPDATE_EVERY == 0:
                    self.value_state = self.value_state.replace(target_params=optax.incremental_update(
                        self.value_state.params, self.value_state.target_params, 0.9))
                episode_loss += loss_values
                self.updates += 1
        gc.collect()
        jax.clear_caches()
        self.counter += 1
        return episode_loss, episode_rewards


class DuelingDQNMax:
    def __init__(self, env, num_actions, observation_shape, seed=0) -> None:
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_actions = num_actions
        self.observation_shape = observation_shape
        self.env = env

        self.value = ValueNetworkMax(num_actions)
        self.value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=self.value.init(self.rng, jnp.ones(observation_shape)),
            target_params=self.value.init(
                self.rng, jnp.ones(observation_shape)),
            tx=optax.adam(learning_rate=ALPHA)
        )
        self.value.apply = jax.jit(self.value.apply)
        self.value_state = self.value_state.replace(target_params=optax.incremental_update(
            self.value_state.params, self.value_state.target_params, 0.9))
        self.replay_buffer = ReplayBuffer(CAPACITY)
        self.counter = 1
        self.updates = 1

    @functools.partial(jax.jit, static_argnums=(0,))
    def policy_greedy(self, q_state, state):
        q_values = self.value.apply(q_state.params, state)
        action = q_values.argmax(axis=-1)[0]
        return action

    def sample(self, state, epsilon=0.1):
        if np.random.uniform() < epsilon:
            return np.random.randint(0, self.num_actions)
        action = np.array(self.policy_greedy(self.value_state, state))
        return action

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, value_state, states, actions, rewards, next_states,  dones):
        value_next_target = self.value.apply(
            value_state.target_params, next_states)
        value_next_target = jnp.max(value_next_target, axis=-1)
        next_q_value = (rewards + (1 - dones) * GAMMA * value_next_target)

        def mse_loss(params):
            value_pred = self.value.apply(params, states)
            value_pred = value_pred[jnp.arange(
                value_pred.shape[0]), actions.squeeze()]
            return ((jax.lax.stop_gradient(next_q_value) - value_pred) ** 2).mean()

        loss_value, grads = jax.value_and_grad(
            mse_loss)(value_state.params)
        value_state = value_state.apply_gradients(grads=grads)
        return loss_value, value_state

    def train_single_step(self):
        state = self.env.reset(seed=self.seed)[0]
        key = self.rng
        epsilon = linear_schedule(
            start_e=1, end_e=0.05, duration=500, t=0 if self.counter <= 50 else self.counter-50)
        episode_loss, episode_rewards = 0, 0
        for _ in range(500):
            action = self.sample(np.expand_dims(state, axis=0), epsilon)
            next_state, reward, done, truncated, info = self.env.step(action)
            self.replay_buffer.push(
                [state, action, reward, next_state, done or truncated])
            state = next_state
            episode_rewards += reward

            if truncated or done:
                break

            if len(self.replay_buffer.buffer) > 128:
                indices = self.replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = self.replay_buffer.get_batch(
                    indices)
                loss_values, self.value_state = self.update(self.value_state,
                                                            states, actions, rewards, next_states,  dones)
                if self.updates % UPDATE_EVERY == 0:
                    self.value_state = self.value_state.replace(target_params=optax.incremental_update(
                        self.value_state.params, self.value_state.target_params, 0.9))
                episode_loss += loss_values
                self.updates += 1
        gc.collect()
        jax.clear_caches()
        self.counter += 1
        return episode_loss, episode_rewards


class Simulation:
    def __init__(self, env_name, algorithm) -> None:
        self.env_name = env_name
        self.algorithm = algorithm
        self.env = gym.make(self.env_name)
        self.num_actions = self.env.action_space.n
        self.observation_shape = self.env.observation_space.shape

    def train(self, num_avg=5, episodes=1000):
        self.losses, self.rewards = np.zeros(
            (num_avg, episodes)), np.zeros((num_avg, episodes))

        for seed in range(num_avg):
            self.algo = self.algorithm(
                self.env, self.num_actions, self.observation_shape, seed=seed)
            for ep in tqdm.tqdm(range(1, episodes+1)):
                loss, reward = self.algo.train_single_step()
                self.losses[seed][ep-1] = loss
                self.rewards[seed][ep-1] = reward


if __name__ == '__main__':
    # cartpole_dqn_mean = Simulation('CartPole-v1', algorithm=DuelingDQNMean)
    # cartpole_dqn_mean.train()
    # rewards_cartpole_dqn_mean = cartpole_dqn_mean.rewards
    # mean_rcr = np.mean(rewards_cartpole_dqn_mean, axis=0)
    # std_rcr = np.std(rewards_cartpole_dqn_mean, axis=0)
    # plot_data(mean_rcr, std_rcr, name='Cartpole dqn_mean_try')

    cartpole_dqn_max = Simulation('CartPole-v1', algorithm=DuelingDQNMax)
    cartpole_dqn_max.train()
    rewards_cartpole_dqn_max = cartpole_dqn_max.rewards
    mean_rcb = np.mean(rewards_cartpole_dqn_max, axis=0)
    std_rcb = np.std(rewards_cartpole_dqn_max, axis=0)
    plot_data(mean_rcb, std_rcb, name='Cartpole dqn_max')

    # acrobot_dqn_mean = Simulation('Acrobot-v1', algorithm=DuelingDQNMean)
    # acrobot_dqn_mean.train()
    # rewards_acrobot_dqn_mean = acrobot_dqn_mean.rewards
    # mean_rar = np.mean(rewards_acrobot_dqn_mean, axis=0)
    # std_rar = np.std(rewards_acrobot_dqn_mean, axis=0)
    # plot_data(mean_rar, std_rar, name='Acrobot dqn_mean')

    acrobot_dqn_max = Simulation('Acrobot-v1', algorithm=DuelingDQNMax)
    acrobot_dqn_max.train()
    rewards_acrobot_dqn_max = acrobot_dqn_max.rewards
    mean_rab = np.mean(rewards_acrobot_dqn_max, axis=0)
    std_rab = np.std(rewards_acrobot_dqn_max, axis=0)
    plot_data(mean_rab, std_rab, name='Acrobot dqn_max')

    # mean_mat = [mean_rcr, mean_rcb, mean_rar, mean_rab]

    # std_mat = [std_rcr, std_rcb, std_rar, std_rab]

    # names = ['cartpole_dqn_mean', 'cartpole_dqn_max',
    #          'acrobot_dqn_mean', 'acrobot_dqn_max']

    # fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    # for i in range(2):
    #     for j in range(2):
    #         mean = mean_mat[i*2+j]
    #         std = std_mat[i*2+j]
    #         x = range(len(mean))
    #         ax[i][j].plot(x, mean, color='blue', label='Mean')
    #         ax[i][j].plot(x, smooth_rewards(mean),
    #                       color='orange', label='smoothed')
    #         ax[i][j].fill_between(x, mean - std, mean + std, color='blue',
    #                               alpha=0.3, label='Mean ± Std')
    #         ax[i][j].set_xlabel('Steps')
    #         ax[i][j].set_ylabel('Rewards')
    #         ax[i][j].set_title(names[i*2+j])
    #         ax[i][j].legend()
    #         ax[i][j].grid(True)

    # plt.savefig("FullDataDQN.png")
    # plt.show()
