import os
import tqdm
import matplotlib.pyplot as plt

from strategy import Strategies
from room_environment import *


class Solver:
    def __init__(self, env, type, size, sparsity, episode_length) -> None:
        self.type = type
        self.size = size
        self.sparsity = sparsity
        self.episode_length = episode_length
        # self.env = Environment(type=type, sparsity=sparsity, size=size)
        self.env = env
        self.strategy = Strategies()
        self.num_images = 24

    def initialize(self):
        self.env.reset()
        self.A_ts = []
        self.B_ts = []
        self.all_possible_actions = np.array(
            [self.env._action_vector(i) for i in range(self.env.total_actions)])

    def action_select(self, theta_hat):
        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        return jnp.argmin(jnp.dot(self.all_possible_actions, eig_vecs[0]))

    def simulate(self, strategy='without_reg', plot_images=True):
        self.initialize()
        action = np.random.choice(self.size**4)
        true_map = self.env.map_star
        pred_images = []
        for t in tqdm.tqdm(range(self.episode_length)):
            A_t, B_t = self.env.step(action)
            self.A_ts.append(A_t)
            self.B_ts.append(B_t)

            if strategy == 'without_reg':
                theta_hat = self.strategy.without_regularization(
                    self.A_ts, self.B_ts)
            elif strategy == 'with_reg':
                theta_hat = self.strategy.with_regularization(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.env.sparsity)
            elif strategy == 'with_reg_and_zero_prior':
                theta_hat = self.strategy.with_regularization_and_zero_prior(
                    self.A_ts, self.B_ts, size=self.size)
            elif strategy == 'tikh_reg':
                theta_hat = self.strategy.with_tikhonov_regularization(
                    self.A_ts, self.B_ts, size=self.size)
            elif strategy == 'nnls':
                theta_hat = self.strategy.with_non_negative_least_square(
                    self.A_ts, self.B_ts)
            elif strategy == 'tikh_nnls':
                theta_hat = self.strategy.with_tikhonov_and_nnls(
                    self.A_ts, self.B_ts)
            action = self.action_select(theta_hat)

            if (t+1) % (self.episode_length//self.num_images) == 0:
                pred_images.append(theta_hat.reshape(self.size, self.size))

        pred_images[-1] = theta_hat.reshape(self.size, self.size)
        if plot_images:
            os.makedirs('plots_'+strategy, exist_ok=True)
            fig, ax = plt.subplots(5, 5, figsize=(self.size*2, self.size*2))
            fig.tight_layout()
            for i in range(5):
                for j in range(5):
                    ax[i, j].axis('off')
                    ax[i, j].imshow(pred_images[4*i+j])
                    if i == 0 and j == 0:
                        ax[i, j].set_title('True Map')
                        ax[i, j].imshow(true_map)
                    elif i == 4 and j == 4:
                        ax[i, j].set_title(
                            'Predicted Map at epoch ' + str(self.episode_length))
                    else:
                        ax[i, j].set_title('Predicted Map at epoch ' +
                                           str((self.episode_length//self.num_images)*(4*i + j)))
            # plt.savefig(f'plots/with_{strategy}_{self.type}.png')
            plt.savefig(
                f'plots_{strategy}/with_{strategy}_{self.type}.png')
            plt.close()
        true_map = self.env.map_star
        pred_map = theta_hat.reshape(self.size, self.size)
        return true_map, pred_map


if __name__ == '__main__':
    num_maps = 10

    strategies = ['without_reg', 'with_reg',
                  'with_reg_and_zero_prior', 'tikh_reg', 'nnls', 'tikh_nnls']
    map_dict = {}
    evaluation_dict = {}
    for strategy in strategies:
        map_dict[strategy] = []

    metrics = [calculate_psnr, calculate_mse, calculate_ssim]
    for metric in metrics:
        evaluation_dict[metric.__name__] = {}
        for strategy in strategies:
            evaluation_dict[metric.__name__][strategy] = []

    for type in range(num_maps):
        rng = np.random.default_rng(type)
        environ = CreateRooms(type=type, size=12)
        solve = Solver(environ, type=type, size=12,
                       sparsity=0.6, episode_length=1000)
        for strategy in strategies:
            true_map, pred_map = solve.simulate(strategy)
            map_dict[strategy].append((true_map, pred_map))

    for metric in metrics:
        for strategy in strategies:
            evaluation_dict[metric.__name__][strategy] += list(
                map(lambda img: metric(img[0], img[1]), map_dict[strategy]))

    fig, ax = plt.subplots(1, len(metrics), figsize=(17, 5))

    for ind, func in enumerate(metrics):
        for strategy in strategies:
            ax[ind].plot(evaluation_dict[func.__name__]
                         [strategy], label=strategy)
        ax[ind].set_xlabel('ith Image')
        ax[ind].set_ylabel(func.__name__.upper())
        ax[ind].legend()

    write_value(evaluation_dict)
    plt.savefig('plots/metrics.png', bbox_inches='tight')
    plt.close()
