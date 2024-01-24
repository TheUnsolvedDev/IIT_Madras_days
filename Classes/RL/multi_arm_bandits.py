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
    best_action_count = 0
    best_action_count_history = np.zeros(NUM_STEPS)
    Q = np.zeros(NUM_ACTIONS)
    N = np.zeros(NUM_ACTIONS)

    for t in range(NUM_STEPS):
        if np.random.uniform(0, 1) <= epsilon:
            action = np.random.randint(0, NUM_ACTIONS)
        else:
            action = np.argmax(Q)
        if action == np.argmax(MUs):
            best_action_count += 1
        best_action_count_history[t] = best_action_count
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1/N[action])*(reward-Q[action])
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1]*(t+1)+reward)/(t+1)
    return Q, rewards, average_rewards, best_action_count_history


def softmax(x):
    value = np.exp(x)
    return value/np.sum(value, keepdims=True)


def softmax_policy(bandit, temparature=2):
    rewards = np.zeros(NUM_STEPS)
    average_rewards = np.zeros(NUM_STEPS)
    best_action_count = 0
    best_action_count_history = np.zeros(NUM_STEPS)
    Q = np.zeros(NUM_ACTIONS)
    N = np.zeros(NUM_ACTIONS)

    for t in range(NUM_STEPS):
        action = np.random.choice(
            NUM_ACTIONS, 1, p=softmax(Q/temparature)).squeeze()
        if action == np.argmax(MUs):
            best_action_count += 1
        best_action_count_history[t] = best_action_count
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1/N[action])*(reward-Q[action])
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1]*(t+1)+reward)/(t+1)
    return Q, rewards, average_rewards, best_action_count_history


def upper_confidence_bound_policy(bandit, c=2):
    rewards = np.zeros(NUM_STEPS)
    average_rewards = np.zeros(NUM_STEPS)
    best_action_count = 0
    best_action_count_history = np.zeros(NUM_STEPS)
    Q = np.zeros(NUM_ACTIONS)
    N = np.zeros(NUM_ACTIONS)

    for t in range(NUM_STEPS):
        action = np.argmax(Q + c * np.sqrt(np.log(t + 1) / (N + 1e-8)))
        if action == np.argmax(MUs):
            best_action_count += 1
        best_action_count_history[t] = best_action_count
        reward = bandit.step(action)
        N[action] += 1
        Q[action] += (1/N[action]) * (reward - Q[action])
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1] *
                              (t + 1) + reward) / (t + 1)
    return Q, rewards, average_rewards, best_action_count_history


def bayesian_ucb_policy(bandit, alpha=1, beta=1):
    rewards = np.zeros(NUM_STEPS)
    average_rewards = np.zeros(NUM_STEPS)
    best_action_count = np.zeros(NUM_STEPS)
    posterior_alpha_beta = np.ones((NUM_ACTIONS, 2))

    for t in range(NUM_STEPS):
        ucb_values = posterior_alpha_beta[:, 0] / \
            (posterior_alpha_beta[:, 0] + posterior_alpha_beta[:, 1])
        ucb_values += np.sqrt(alpha * np.log(t + 1) /
                              (posterior_alpha_beta[:, 0] + posterior_alpha_beta[:, 1]))

        action = np.argmax(ucb_values)
        if action == np.argmax(MUs):
            best_action_count[t] = best_action_count[t-1] + 1
        reward = bandit.step(action)
        posterior_alpha_beta[action, 0] += reward
        posterior_alpha_beta[action, 1] += 1 - reward
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1] *
                              (t + 1) + reward) / (t + 1)

    return posterior_alpha_beta[:, 0] / np.sum(posterior_alpha_beta, axis=1), rewards, average_rewards, best_action_count


def probably_approximately_correct_policy(bandit):
    return


def median_eliminate_policy(bandit):
    return


def thompson_sampling_policy(bandit, alpha=1, beta=1):
    rewards = np.zeros(NUM_STEPS)
    average_rewards = np.zeros(NUM_STEPS)
    best_action_count = np.zeros(NUM_STEPS)
    alpha_beta = np.ones((NUM_ACTIONS, 2))

    for t in range(NUM_STEPS):
        sampled_means = np.random.beta(alpha_beta[:, 0], alpha_beta[:, 1])
        action = np.argmax(sampled_means)
        if action == np.argmax(MUs):
            best_action_count[t] = best_action_count[t-1] + 1
        reward = bandit.step(action)
        alpha_beta[action, 0] += reward
        alpha_beta[action, 1] += 1 - reward
        rewards[t] = reward
        average_rewards[t] = (average_rewards[t-1] *
                              (t + 1) + reward) / (t + 1)
    return alpha_beta[:, 0] / np.sum(alpha_beta, axis=1), rewards, average_rewards, best_action_count


def main():
    bandit = Bandits()
    settings = {
        'greedy': epsilon_greedy_policy(bandit, epsilon=0),
        'epsilon_greedy': epsilon_greedy_policy(bandit),
        'softmax_policy': softmax_policy(bandit),
        'upper_confidence_bound_policy': upper_confidence_bound_policy(bandit),
        'bayesian_ucb_policy': bayesian_ucb_policy(bandit),
        # 'thompson_sampling_policy': thompson_sampling_policy(bandit),
    }

    fig_cols = len(settings.keys())
    fig_rows = 3
    fig, ax = plt.subplots(
        fig_cols, fig_rows, figsize=(fig_cols*2, fig_rows*3))
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
