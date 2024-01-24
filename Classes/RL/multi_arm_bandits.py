import numpy as np
import matplotlib.pyplot as plt

NUM_STEPS = 250
MUs = np.array([5, -5, 6, 10, -7])
NUM_ACTIONS = len(MUs)


class Bandits:
    def __init__(self) -> None:
        self.mu_values = MUs

    def step(self, action):
        return np.random.normal(loc=self.mu_values[action], scale=0.1)


def epsilon_greedy_policy(bandit, epsilon=0.5):
    rewards = np.zeros(NUM_STEPS)
    average_rewards = np.zeros(NUM_STEPS)
    best_action_count = np.zeros(NUM_STEPS)
    Q = np.zeros(NUM_ACTIONS)
    N = np.zeros(NUM_ACTIONS)

    for t in range(NUM_STEPS):
        if np.random.uniform(0, 1) <= epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(Q)
        if action == np.argmax(MUs):
            best_action_count[t] = best_action_count[t-1] + 1
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1/N[action])*(reward-Q[action])
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1]*(t+1)+reward)/(t+1)
    return Q, rewards, average_rewards, best_action_count


def softmax(x):
    value = np.exp(x)
    return value/np.sum(value, keepdims=True)


def softmax_policy(bandit, temparature=2):
    rewards = np.zeros(NUM_STEPS)
    average_rewards = np.zeros(NUM_STEPS)
    best_action_count = np.zeros(NUM_STEPS)
    Q = np.zeros(NUM_ACTIONS)
    N = np.zeros(NUM_ACTIONS)

    for t in range(NUM_STEPS):
        action = np.random.choice(
            NUM_ACTIONS, 1, p=softmax(Q/temparature)).squeeze()
        if action == np.argmax(MUs):
            best_action_count[t] = best_action_count[t-1] + 1
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1/N[action])*(reward-Q[action])
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1]*(t+1)+reward)/(t+1)
    return Q, rewards, average_rewards, best_action_count


def main():
    bandit = Bandits()
    settings = {
        'greedy': epsilon_greedy_policy(bandit, epsilon=0),
        'epsilon_greedy': epsilon_greedy_policy(bandit),
        'softmax_policy': softmax_policy(bandit)
    }

    fig, ax = plt.subplots(len(settings.keys()), 3, figsize=(12, 9))
    for ind, key in enumerate(settings.keys()):
        fig.tight_layout()
        ax[ind][0].set_title(key)
        ax[ind][0].plot(settings[key][1])
        ax[ind][0].set_xlabel('steps')
        ax[ind][0].set_ylabel('rewards')

        ax[ind][1].plot(settings[key][3]/NUM_STEPS)
        ax[ind][1].set_xlabel('step')
        ax[ind][1].set_ylabel('% of optimal action')

        ax[ind][2].plot(settings[key][2])
        ax[ind][2].set_xlabel('steps')
        ax[ind][2].set_ylabel('average rewards')
        print(key, 'Best arm:', np.argmax(settings[key][0]))

    plt.show()


if __name__ == '__main__':
    main()
