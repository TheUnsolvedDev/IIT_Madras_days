import os
import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import gc

from strategy import *
from environment import *
from utils import *

import argparse


class Solver:
    def __init__(self,
                 env: 'RoomEnvironment',  # type: ignore
                 type: int,
                 size: int,
                 sparsity: int,
                 episode_length: int) -> None:
        """
        Args:
            env (RoomEnvironment): The environment to use for the simulation.
            type (int): The seed to use for generating the room.
            size (int): The size of the room.
            sparsity (int): The sparsity of the room.
            episode_length (int): The length of each episode.
        """
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

    def initialize(self) -> None:
        """
        Resets the environment and clears the collected data.

        Returns:
            None
        """
        self.env.reset()
        self.A_ts = []
        self.B_ts = []

    def simulate(self,
                 reconstruction: str = 'without_regularization',
                 strategy: str = 'min_eigenvalue_info',
                 lamda: float = 0.01,
                 plot_images: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the environment using the provided reconstruction method and strategy.

        Args:
            reconstruction (str): The reconstruction method to use. Currently supports 'without_regularization', 'with_tikhonov_and_identity',
            'with_tikhonov_and_lambda_identity', 'with_LOG_regularization_and_sparsity_prior', 'with_LOG_regularization_and_zero_prior', 'with_Laplacian_regularization',
            and 'with_non_negative_least_square'. Defaults to 'without_regularization'.
            strategy (str): The strategy to use. Currently supports 'min_eigenvalue_info' and 'random'. Defaults to 'min_eigenvalue_info'.
            lamda (float): The regularization parameter. Defaults to 0.01.
            plot_images (bool): Whether to save plots of the predicted maps. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The true map and the predicted map.
        """
        self.initialize()
        action = np.random.choice(self.env.total_actions)
        true_map = self.env.map_star
        pred_images = []

        os.makedirs(f'{strategy}_{size}x{size}/plots_{reconstruction}', exist_ok=True)
        logger = DataLogger(
            file_path=f'{strategy}_{size}x{size}/plots_{reconstruction}/{reconstruction}_{strategy}_{self.type}.csv')
        action_hash = defaultdict(int)

        for t in tqdm.tqdm(range(self.episode_length)):
            A_t, B_t = self.env.step(action)
            self.A_ts.append(A_t)
            self.B_ts.append(B_t)

            if reconstruction == 'without_regularization_and_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0, lamda=0.0)
            elif reconstruction == 'without_regularization_and_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.0)

            elif reconstruction == 'with_tikhonov_and_identity_and_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0, lamda=1)
            elif reconstruction == 'with_tikhonov_and_identity_and_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=1)

            elif reconstruction == 'with_tikhonov_and_lambda_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0, lamda=0.01)
            elif reconstruction == 'with_tikhonov_and_lambda_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_tikhonov_and_lambda_identity(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.01)

            elif reconstruction == 'with_Gauss_regularization_and_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_Gauss_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0, lamda=1)
            elif reconstruction == 'with_Gauss_regularization_and_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_Gauss_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=1)

            elif reconstruction == 'with_Gauss_regularization_and_lambda_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_Gauss_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0, lamda=0.01)
            elif reconstruction == 'with_Gauss_regularization_and_lambda_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_Gauss_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.01)

            elif reconstruction == 'with_LOG_regularization_and_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_LOG_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0.0, lamda=0.01)
            elif reconstruction == 'with_LOG_regularization_and_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_LOG_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.01)

            elif reconstruction == 'with_LOG_regularization_and_lambda_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_LOG_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0.0, lamda=0.01)
            elif reconstruction == 'with_LOG_regularization_and_lambda_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_LOG_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.01)

            elif reconstruction == 'with_Sharpen_regularization_and_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_Sharpen_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0.0, lamda=0.01)
            elif reconstruction == 'with_Sharpen_regularization_and_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_Sharpen_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.01)

            elif reconstruction == 'with_Sharpen_regularization_and_lambda_zero_prior':
                theta_hat, meta_data = self.reconstruction.with_Sharpen_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=0.0, lamda=0.01)
            elif reconstruction == 'with_Sharpen_regularization_and_lambda_sparsity_prior':
                theta_hat, meta_data = self.reconstruction.with_Sharpen_regularization_and_sparsity_prior(
                    self.A_ts, self.B_ts, size=self.size, sparsity=self.sparsity, lamda=0.01)

            elif reconstruction == 'with_Laplacian_regularization':
                theta_hat, meta_data = self.reconstruction.with_Laplacian_regularization(
                    self.A_ts, self.B_ts, size=self.size)
            elif reconstruction == 'with_non_negative_least_square':
                theta_hat, meta_data = self.reconstruction.with_non_negative_least_square(
                    self.A_ts, self.B_ts)
            else:
                print("Wrong reconstruction choosen", reconstruction)
                exit(0)
            kernel, lamda = meta_data
            V_t = jnp.dot(
                jnp.array(self.A_ts).T, jnp.array(self.A_ts))
            if kernel != None:
                V_t += lamda*kernel
            if strategy == 'min_eigenvalue_info':
                action = int(
                    self.policy.min_eigenvalue_info_action(V_t))
            elif strategy == 'random':
                action = self.policy.random_action(V_t)

            if (t+1) % (self.episode_length//self.num_images) == 0:
                pred_images.append(theta_hat.reshape(self.size, self.size))
            rank = jnp.linalg.matrix_rank(jnp.array(self.A_ts))
            pred_map = theta_hat.reshape(true_map.shape)
            mse_value = calculate_mse(true_map, pred_map)
            ssim_value = calculate_ssim(true_map, pred_map)
            psnr_value = calculate_psnr(true_map, pred_map)
            action_vector = convert_actions(action, self.size)
            data = [action, psnr_value, mse_value,
                    ssim_value, rank, action_vector]
            logger.append_log(*data)

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
                f'{strategy}_{size}x{size}/plots_{reconstruction}/with_{reconstruction}_{strategy}_{self.type}.png')
            plt.close()
        true_map = self.env.map_star
        pred_map = theta_hat.reshape(self.size, self.size)
        return true_map, pred_map


if __name__ == '__main__':

    reconstructions_without = sorted([
        'without_regularization_and_zero_prior',
        'without_regularization_and_sparsity_prior'])
    reconstructions_tikhonov = sorted([
        'with_tikhonov_and_identity_and_zero_prior',
        'with_tikhonov_and_identity_and_sparsity_prior',
        'with_tikhonov_and_lambda_zero_prior',
        'with_tikhonov_and_lambda_sparsity_prior'])
    reconstructions_sharpen = sorted([
        'with_Sharpen_regularization_and_zero_prior',
        'with_Sharpen_regularization_and_sparsity_prior',
        'with_Sharpen_regularization_and_lambda_zero_prior',
        'with_Sharpen_regularization_and_lambda_sparsity_prior'])
    reconstructions_LOGian = sorted([
        'with_LOG_regularization_and_zero_prior',
        'with_LOG_regularization_and_sparsity_prior',
        'with_LOG_regularization_and_lambda_zero_prior',
        'with_LOG_regularization_and_lambda_sparsity_prior'])
    reconstructions_gaussian = sorted([
        'with_Gauss_regularization_and_zero_prior',
        'with_Gauss_regularization_and_sparsity_prior',
        'with_Gauss_regularization_and_lambda_zero_prior',
        'with_Gauss_regularization_and_lambda_sparsity_prior'])
    reconstructions_laplacian = ['with_Laplacian_regularization']
    reconstruction_nnls = ['with_non_negative_least_square']

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reconstruction', type=str,
                        default='with_tikhonov_and_lambda_zero_prior')
    parser.add_argument('-st', '--strategy', type=str, default='random')
    parser.add_argument('--lamda', type=float, default=0.01)
    parser.add_argument('--plot_images', type=bool, default=True)
    parser.add_argument('-s', '--size', type=int, default=15)
    parser.add_argument('-nm', '--num_maps', type=int, default=5)
    args = parser.parse_args()

    num_maps = args.num_maps
    size = args.size
    lamda = args.lamda

    if args.reconstruction in reconstructions_without:
        reconstructions = [args.reconstruction]
    elif args.reconstruction in reconstructions_tikhonov:
        reconstructions = [args.reconstruction]
    elif args.reconstruction in reconstructions_sharpen:
        reconstructions = [args.reconstruction]
    elif args.reconstruction in reconstructions_LOGian:
        reconstructions = [args.reconstruction]
    elif args.reconstruction in reconstructions_gaussian:
        reconstructions = [args.reconstruction]
    elif args.reconstruction in reconstructions_laplacian:
        reconstructions = [args.reconstruction]
    else:
        reconstructions = reconstructions_without+reconstructions_tikhonov+reconstructions_sharpen + \
            reconstructions_LOGian+reconstructions_gaussian+reconstructions_laplacian

    if args.strategy == 'min_eigenvalue_info':
        strategies = ['min_eigenvalue_info']
    elif args.strategy == 'random':
        strategies = ['random']
    else:
        strategies = ['min_eigenvalue_info', 'random']

    map_dict = {}
    evaluation_dict = {}

    for reconstruction in reconstructions:
        map_dict[reconstruction] = []

    metrics = [calculate_psnr, calculate_mse, calculate_ssim]

    for metric in metrics:
        evaluation_dict[metric.__name__] = {}
        for reconstruction in reconstructions:
            evaluation_dict[metric.__name__][reconstruction] = []

    for type in range(num_maps):
        for reconstruction in reconstructions:
            rng = np.random.default_rng(type)
            # environ = CreateRooms(type=type, size=size)
            environ = Environment(size=size,rng=type)
            solve = Solver(environ, type=type, size=size,
                           sparsity=0.6, episode_length=int(1.5*size*size))
            print('Working on ' + reconstruction +
                  ' with type ' + str(type), end='')
            for strategy in strategies:
                print(' and ' + strategy, end='\n')
                true_map, pred_map = solve.simulate(
                    reconstruction=reconstruction, strategy=strategy)
                map_dict[reconstruction].append((true_map, pred_map))
                jax.clear_caches()
                gc.collect()
            jax.clear_caches()
            gc.collect()
        jax.clear_caches()
        gc.collect()    

    for metric in metrics:
        for reconstruction in reconstructions:
            evaluation_dict[metric.__name__][reconstruction] += list(
                map(lambda img: metric(img[0], img[1]), map_dict[reconstruction]))

    fig, ax = plt.subplots(1, len(metrics), figsize=(17, 5))

    for ind, func in enumerate(metrics):
        for reconstruction in reconstructions:
            ax[ind].plot(evaluation_dict[func.__name__]
                         [reconstruction], label=reconstruction)
        ax[ind].set_xlabel('ith Image')
        ax[ind].set_ylabel(func.__name__.upper())
        ax[ind].legend()

    write_value(evaluation_dict)
    plt.savefig('plots/metrics.png', bbox_inches='tight')
    plt.close()
