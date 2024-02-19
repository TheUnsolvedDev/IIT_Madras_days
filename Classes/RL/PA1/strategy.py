import numpy as np


class Policy:
    @staticmethod
    def softmax(Q, state):
        logits = Q[state] - Q[state].max()
        probs = np.exp(logits)/np.sum(np.exp(logits), axis=0, keepdims=True)
        action = np.random.choice(len(probs), p=probs)
        return action

    @staticmethod
    def random(Q, state):
        action = np.random.choice(Q[state])
        return action

    @staticmethod
    def epsilon_greedy(Q, state, epsilon=0.1):
        if np.random.uniform() < epsilon:
            action = np.random.choice(len(Q[state]))
        else:
            action = np.argmax(Q[state])
        return action


class Agent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, policy=Policy.softmax) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.policy = policy

    def act(self, state, **kwargs):
        if 'epsilon' in kwargs.keys():
            pass
        else:
            return self.policy(self.Q, state)


class QLearning(Agent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, policy=Policy.softmax) -> None:
        super().__init__(num_states, num_actions, alpha, gamma, policy)

    def update(self, state, action, reward, next_state):
        self.Q[state][action] += self.alpha*(reward + self.gamma *
                                             np.max(self.Q[next_state]) - self.Q[state][action])


class SARSA(Agent):
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, policy=Policy.softmax) -> None:
        super().__init__(num_states, num_actions, alpha, gamma, policy)

    def update(self, state, action, reward, next_state, next_action):
        self.Q[state][action] += self.alpha*(reward + self.gamma *
                                             self.Q[next_state][next_action] - self.Q[state][action])
