import numpy as np
import matplotlib.pyplot as plt
import tqdm

from environment import *
from strategy import *


def create_environment():
    num_cols = 10
    num_rows = 10
    obstructions = np.array([[0, 7], [1, 1], [1, 2], [1, 3], [1, 7], [2, 1], [2, 3],
                            [2, 7], [3, 1], [3, 3], [3, 5], [
                                4, 3], [4, 5], [4, 7],
                            [5, 3], [5, 7], [5, 9], [6, 3], [
                                6, 9], [7, 1], [7, 6],
                            [7, 7], [7, 8], [7, 9], [8, 1], [8, 5], [8, 6], [9, 1]])
    bad_states = np.array([[1, 9], [4, 2], [4, 4], [7, 5], [9, 9]])
    restart_states = np.array([[3, 7], [8, 2]])
    start_state = np.array([[3, 6]])
    goal_states = np.array([[0, 9], [2, 2], [8, 7]])

    # create model
    gw = GridWorld(num_rows=num_rows,
                   num_cols=num_cols,
                   start_state=start_state,
                   goal_states=goal_states, wind=False)
    gw.add_obstructions(obstructed_states=obstructions,
                        bad_states=bad_states,
                        restart_states=restart_states)
    gw.add_rewards(step_reward=-1,
                   goal_reward=10,
                   bad_state_reward=-6,
                   restart_state_reward=-100)
    gw.add_transition_probability(p_good_transition=0.7,
                                  bias=0.5)
    env = gw.create_gridworld()

    # 0 -> UP, 1-> DOWN, 2 -> LEFT, 3-> RIGHT
    print("Number of actions", env.num_actions)
    print("Number of states", env.num_states)
    print("start state", env.start_state_seq)
    print("goal state(s)", env.goal_states_seq)
    return env


def simulate(env, num_games=1000, max_episodes=500):
    epsilon = 0.1
    game_histories = {}
    agent = QLearning(env.num_states, env.num_actions,
                      policy=Policy.softmax)
    game_histories = {
        'max_rewards': np.zeros(num_games),
        'avg_rewards': np.zeros(num_games),
        'min_rewards': np.zeros(num_games)
    }

    for game in tqdm.tqdm(range(1, num_games+1)):
        state = env.reset()
        stored_rewards, maxm, minm = 0, -np.inf, np.inf

        for ep in range(1, max_episodes+1):
            if state in env.goal_states:
                break
            action = agent.act(state)
            next_state, reward = env.step(state, action)
            agent.update(state, action, reward, next_state)
            state = next_state

            maxm = max(reward, maxm)
            minm = min(reward, minm)
            stored_rewards += reward

        game_histories['min_rewards'][game-1] = minm
        game_histories['avg_rewards'][game-1] = stored_rewards/ep
        game_histories['max_rewards'][game-1] = maxm
        V = np.sum(agent.Q, axis=1).reshape(10, 10)

    return game_histories


def plot_Q(Q):
    V = np.sum(Q, axis=0)
    V.reshape(10, 10)
    plt.imshow(V)
    plt.colorbar()
    plt.show()


def cumulative_mean(data):
    cum_sum = 0
    cum_mean_values = []

    for i, value in enumerate(data, start=1):
        cum_sum += value
        cum_mean = cum_sum / i
        cum_mean_values.append(cum_mean)

    return cum_mean_values


def plot_game_history(game_history):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(game_history['min_rewards'], label='min_rewards')
    ax[0].plot(cumulative_mean(game_history['min_rewards']),
               label='min_rewards')
    ax[0].set_xlabel('steps')
    ax[0].set_ylabel('rewards')
    ax[0].legend()
    ax[1].plot(game_history['avg_rewards'], label='avg_rewards')
    ax[1].plot(cumulative_mean(game_history['avg_rewards']),
               label='avg_rewards')
    ax[1].set_xlabel('steps')
    ax[1].set_ylabel('rewards')
    ax[1].legend()
    ax[2].plot(game_history['max_rewards'], label='max_rewards')
    ax[2].plot(cumulative_mean(game_history['max_rewards']),
               label='max_rewards')
    ax[2].set_xlabel('steps')
    ax[2].set_ylabel('rewards')
    ax[2].legend()
    plt.show()


def main():
    setup = {
        'alpha': [1, 0.1, 0.01],
        'gamma': [0.95, 0.995],
        'policy': [Policy.epsilon_greedy]
    }


if __name__ == '__main__':
    env = create_environment()
    game_history = simulate(env)
    plot_game_history(game_history)
