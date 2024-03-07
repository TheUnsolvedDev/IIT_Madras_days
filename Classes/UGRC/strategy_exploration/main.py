import os
import tqdm
import matplotlib.pyplot as plt

from strategy import *
from room_environment import *
from utils import *


class Solver:
    def __init__(self, env, type, size, sparsity, episode_length) -> None:
        self.type = type
        self.size = size
        self.sparsity = sparsity
        self.episode_length = episode_length
        self.env = env
        self.all_possible_actions = np.array(
            [self.env._action_vector(i) for i in range(self.env.total_actions)])

        self.reconstruction = StrategiesReconstruction()
        self.policy = StartegiesAction(self.all_possible_actions)
        self.num_images = 24

    def initialize(self):
        self.env.reset()
        self.A_ts = []
        self.B_ts = []

    def simulate(self, reconstruction='without_regularization', strategy='min_eigenvalue_info', plot_images=True):
        self.initialize()
        action = np.random.choice(self.size**4)
        true_map = self.env.map_star
        pred_images = []

        os.makedirs(f'plots_{reconstruction}_{strategy}', exist_ok=True)
        logger = DataLogger(
            file_path=f'plots_{reconstruction}_{strategy}/{reconstruction}_{strategy}_{self.type}.csv')

        for t in tqdm.tqdm(range(self.episode_length)):
            A_t, B_t = self.env.step(action)
            self.A_ts.append(A_t)
            self.B_ts.append(B_t)

            if reconstruction == 'without_regularization':
                theta_hat = self.reconstruction.without_regularization(
                    self.A_ts, self.B_ts)
            elif reconstruction == 'with_tikhonov_and_identity':
                theta_hat = self.reconstruction.with_tikhonov_and_identity(
                    self.A_ts, self.B_ts)
            elif reconstruction == 'with_tikhonov_and_lambda_identity':
                theta_hat = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, lamda=0.01)
            elif reconstruction == 'with_LOG_regularization_and_sparsity_prior':
                theta_hat = self.reconstruction.with_LOG_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity)
            elif reconstruction == 'with_LOG_regularization_and_zero_prior':
                theta_hat = self.reconstruction.with_LOG_regularization_and_zero_prior(
                    self.A_ts, self.B_ts)
            elif reconstruction == 'with_Laplacian_regularization':
                theta_hat = self.reconstruction.with_Laplacian_regularization(
                    self.A_ts, self.B_ts, size=self.size)
            elif reconstruction == 'with_non_negative_least_square':
                theta_hat = self.reconstruction.with_non_negative_least_square(
                    self.A_ts, self.B_ts)
            else:
                print("Wrong reconstruction choosen")
                exit(0)

            if strategy == 'min_eigenvalue_info':
                action = self.policy.min_eigenvalue_info_action(theta_hat)
            else:
                action = self.policy.random_action(theta_hat)

            if (t+1) % (self.episode_length//self.num_images) == 0:
                pred_images.append(theta_hat.reshape(self.size, self.size))

            pred_map = theta_hat.reshape(true_map.shape)
            mse_value = calculate_mse(true_map, pred_map)
            ssim_value = calculate_ssim(true_map, pred_map)
            psnr_value = calculate_psnr(true_map, pred_map)
            data = [action, psnr_value, mse_value, ssim_value]
            logger.append_log(data)

        pred_images[-1] = theta_hat.reshape(self.size, self.size)
        if plot_images:
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
                f'plots_{reconstruction}_{strategy}/with_{reconstruction}_{strategy}_{self.type}.png')
            plt.close()
        true_map = self.env.map_star
        pred_map = theta_hat.reshape(self.size, self.size)
        return true_map, pred_map


if __name__ == '__main__':
    num_maps = 10

    strategies = ['without_regularization', 'with_tikhonov_and_identity',
                  'with_tikhonov_and_lambda_identity', 'with_LOG_regularization_and_sparsity_prior', 'with_LOG_regularization_and_zero_prior', 'with_Laplacian_regularization', 'with_non_negative_least_square']
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
            true_map, pred_map = solve.simulate(reconstruction=strategy,strategy = 'random')
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
