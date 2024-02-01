import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
import os
from scipy.optimize import nnls
import pylops

from environment import *
from utils import *

NUM_IMAGES = 24


def strategy1(SPARSITY, TYPE, SIZE, EPISODE_LENGTH):
    # without regularization
    env = Environment(type=TYPE, sparsity=SPARSITY, size=SIZE)
    env.reset()

    A_ts, B_ts = [], []
    action = np.random.choice(SIZE**4)
    delta = 0.001

    all_possible_actions = np.array(
        [env._action_vector(i) for i in range(env.total_actions)])
    plt.axis('off')

    pred_images = []
    true_map = env.map_star
    for t in tqdm.tqdm(range(EPISODE_LENGTH)):
        A_t, B_t = env.step(action)
        A_ts.append(A_t)
        B_ts.append(B_t)

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        theta_hat_1 = jnp.linalg.pinv(V_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)))
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))

        if (t+1) % (EPISODE_LENGTH//NUM_IMAGES) == 0:
            pred_images.append(theta_hat.reshape(SIZE, SIZE))

    pred_images[-1] = theta_hat.reshape(SIZE, SIZE)
    os.makedirs('plots_without_regualrization', exist_ok=True)
    fig, ax = plt.subplots(5, 5, figsize=(SIZE*2, SIZE*2))
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
                    'Predicted Map at epoch ' + str(EPISODE_LENGTH))
            else:
                ax[i, j].set_title('Predicted Map at epoch ' +
                                   str((EPISODE_LENGTH//NUM_IMAGES)*(4*i + j)))
    plt.savefig(f'plots/without_regularization_{TYPE}.png')
    plt.savefig(
        f'plots_without_regualrization/without_regularization_{TYPE}.png')
    plt.close()
    true_map = env.map_star
    pred_map = theta_hat.reshape(SIZE, SIZE)
    return true_map, pred_map, A_ts, B_ts


def strategy2(SPARSITY, TYPE, SIZE, EPISODE_LENGTH, ZERO_PRIOR=False):
    # with regularization
    env = Environment(type=TYPE, sparsity=SPARSITY, size=SIZE)
    env.reset()

    A_ts, B_ts = [], []
    L = LOG_kernel(SIZE)
    H = jnp.dot(L, L)
    action = np.random.choice(SIZE**4)
    delta = 0.001
    theta_prior = env.sparsity*jnp.ones((SIZE*SIZE, 1))
    if ZERO_PRIOR:
        theta_prior = jnp.zeros((SIZE*SIZE, 1))

    all_possible_actions = np.array(
        [env._action_vector(i) for i in range(env.total_actions)])
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    pred_images = []
    true_map = env.map_star
    for t in tqdm.tqdm(range(EPISODE_LENGTH)):
        A_t, B_t = env.step(action)
        A_ts.append(A_t)
        B_ts.append(B_t)

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = H + V_t
        theta_hat_1 = jnp.linalg.pinv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + H@theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))

        if (t+1) % (EPISODE_LENGTH//NUM_IMAGES) == 0:
            pred_images.append(theta_hat.reshape(SIZE, SIZE))

    pred_images[-1] = theta_hat.reshape(SIZE, SIZE)
    if ZERO_PRIOR:
        os.makedirs('plots_with_priorless_regualrization', exist_ok=True)
    else:
        os.makedirs('plots_with_regualrization', exist_ok=True)

    fig, ax = plt.subplots(5, 5, figsize=(SIZE*2, SIZE*2))
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
                    'Predicted Map at epoch ' + str(EPISODE_LENGTH))
            else:
                ax[i, j].set_title('Predicted Map at epoch ' +
                                   str((EPISODE_LENGTH//NUM_IMAGES)*(4*i + j)))
    if ZERO_PRIOR:
        plt.savefig(f'plots/with_priorless_regularization_{TYPE}.png')
        plt.savefig(
            f'plots_with_priorless_regualrization/with_priorless_regularization_{TYPE}.png')
    else:
        plt.savefig(f'plots/with_regularization_{TYPE}.png')
        plt.savefig(
            f'plots_with_regualrization/with_regularization_{TYPE}.png')

    plt.close()
    true_map = env.map_star
    pred_map = theta_hat.reshape(SIZE, SIZE)
    return true_map, pred_map, A_ts, B_ts

# tikhonov regularization


def strategy3(SPARSITY, TYPE, SIZE, EPISODE_LENGTH):
    env = Environment(type=TYPE, sparsity=SPARSITY, size=SIZE)
    env.reset()

    A_ts, B_ts = [], []
    all_possible_actions = np.array(
        [env._action_vector(i) for i in range(env.total_actions)])
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    action = np.random.choice(SIZE**4)

    pred_images = []
    true_map = env.map_star
    for t in tqdm.tqdm(range(EPISODE_LENGTH)):
        A_t, B_t = env.step(action)
        A_ts.append(A_t)
        B_ts.append(B_t)

        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        theta_hat = pylops.optimization.leastsquares.regularized_inversion(
            projection,
            received,
            [pylops.Laplacian(dims=(SIZE, SIZE), edge=True,
                              weights=(3, 3), dtype="float32")],
            epsRs=[np.sqrt(0.1)],
            **dict(damp=np.sqrt(1e-4), iter_lim=50, show=0)
        )[0].reshape(-1, 1)

        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))

        if (t+1) % (EPISODE_LENGTH//NUM_IMAGES) == 0:
            pred_images.append(theta_hat.reshape(SIZE, SIZE))

    pred_images[-1] = theta_hat.reshape(SIZE, SIZE)
    os.makedirs('plots_with_gen_tikhonov_regualrization', exist_ok=True)

    fig, ax = plt.subplots(5, 5, figsize=(SIZE*2, SIZE*2))
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
                    'Predicted Map at epoch ' + str(EPISODE_LENGTH))
            else:
                ax[i, j].set_title('Predicted Map at epoch ' +
                                   str((EPISODE_LENGTH//NUM_IMAGES)*(4*i + j)))
    plt.savefig(f'plots/with_gen_tikhonov_regularization_{TYPE}.png')
    plt.savefig(
        f'plots_with_gen_tikhonov_regualrization/with_gen_tikhonov_regularization_{TYPE}.png')
    plt.close()
    true_map = env.map_star
    pred_map = theta_hat.reshape(SIZE, SIZE)
    return true_map, pred_map, A_ts, B_ts

# non negative least square

def strategy4(SPARSITY, TYPE, SIZE, EPISODE_LENGTH):
    env = Environment(type=TYPE, sparsity=SPARSITY, size=SIZE)
    env.reset()

    A_ts, B_ts = [], []
    all_possible_actions = np.array(
        [env._action_vector(i) for i in range(env.total_actions)])
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    action = np.random.choice(SIZE**4)

    pred_images = []
    true_map = env.map_star
    for t in tqdm.tqdm(range(EPISODE_LENGTH)):
        A_t, B_t = env.step(action)
        A_ts.append(A_t)
        B_ts.append(B_t)

        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        theta_hat = nnls(projection, received)[0].reshape(-1, 1)

        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))

        if (t+1) % (EPISODE_LENGTH//NUM_IMAGES) == 0:
            pred_images.append(theta_hat.reshape(SIZE, SIZE))

    pred_images[-1] = theta_hat.reshape(SIZE, SIZE)
    os.makedirs('plots_with_non_negative_ls', exist_ok=True)

    fig, ax = plt.subplots(5, 5, figsize=(SIZE*2, SIZE*2))
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
                    'Predicted Map at epoch ' + str(EPISODE_LENGTH))
            else:
                ax[i, j].set_title('Predicted Map at epoch ' +
                                   str((EPISODE_LENGTH//NUM_IMAGES)*(4*i + j)))
    plt.savefig(f'plots/with_non_negative_ls_{TYPE}.png')
    plt.savefig(f'plots_with_non_negative_ls/with_non_negative_ls_{TYPE}.png')
    plt.close()
    true_map = env.map_star
    pred_map = theta_hat.reshape(SIZE, SIZE)
    return true_map, pred_map, A_ts, B_ts

# non negative least square with tikhonov regularization

def strategy5(SPARSITY, TYPE, SIZE, EPISODE_LENGTH, LAMBDA=0.1):
    env = Environment(type=TYPE, sparsity=SPARSITY, size=SIZE)
    env.reset()

    A_ts, B_ts = [], []
    all_possible_actions = np.array(
        [env._action_vector(i) for i in range(env.total_actions)])
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    action = np.random.choice(SIZE**4)

    pred_images = []
    true_map = env.map_star
    for t in tqdm.tqdm(range(EPISODE_LENGTH)):
        A_t, B_t = env.step(action)
        A_ts.append(A_t)
        B_ts.append(B_t)

        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        num_variables = projection.shape[1]

        PROJ_TKNV = np.concatenate([projection,
                                    np.sqrt(LAMBDA) *
                                    np.eye(num_variables)
                                    ])
        RECV_TKNV = np.concatenate(
            [received, np.zeros(num_variables)])
        theta_hat = nnls(PROJ_TKNV, RECV_TKNV)[0].reshape((-1, 1))

        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))

        if (t+1) % (EPISODE_LENGTH//NUM_IMAGES) == 0:
            pred_images.append(theta_hat.reshape(SIZE, SIZE))

    pred_images[-1] = theta_hat.reshape(SIZE, SIZE)
    os.makedirs('plots_with_nnls_tikhonov', exist_ok=True)

    fig, ax = plt.subplots(5, 5, figsize=(SIZE*2, SIZE*2))
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
                    'Predicted Map at epoch ' + str(EPISODE_LENGTH))
            else:
                ax[i, j].set_title('Predicted Map at epoch ' +
                                   str((EPISODE_LENGTH//NUM_IMAGES)*(4*i + j)))
    plt.savefig(f'plots/with_nnls_tikhonov_{TYPE}.png')
    plt.savefig(f'plots_with_nnls_tikhonov/with_nnls_tikhonov_{TYPE}.png')
    plt.close()
    true_map = env.map_star
    pred_map = theta_hat.reshape(SIZE, SIZE)
    return true_map, pred_map, A_ts, B_ts


# def strategy2(SPARSITY, TYPE, SIZE, EPISODE_LENGTH, ZERO_PRIOR=False):
#     # with regularization
#     env = Environment(type=TYPE, sparsity=SPARSITY, size=SIZE)
#     env.reset()

#     A_ts, B_ts = [], []
#     L = LOG_kernel(SIZE)
#     H = jnp.dot(L, L)
#     action = np.random.choice(SIZE**4)
#     delta = 0.001
#     theta_prior = env.sparsity*jnp.ones((SIZE*SIZE, 1))
#     if ZERO_PRIOR:
#         theta_prior = jnp.zeros((SIZE*SIZE, 1))

#     all_possible_actions = np.array(
#         [env._action_vector(i) for i in range(env.total_actions)])
#     fig, ax = plt.subplots(1, 2, figsize=(14, 6))

#     for t in tqdm.tqdm(range(EPISODE_LENGTH)):
#         A_t, B_t = env.step(action)
#         A_ts.append(A_t)
#         B_ts.append(B_t)

#         V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
#         M_t = H + V_t
#         theta_hat_1 = jnp.linalg.pinv(M_t)
#         theta_hat_2 = (jnp.array(A_ts).T @
#                        jnp.array(B_ts).reshape((-1, 1)) + H@theta_prior)
#         theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
#         eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
#         action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))

#         if (t+1) % 100 == 0:
#             true_map = env.map_star
#             pred_map = theta_hat.reshape(SIZE, SIZE)
#             if ZERO_PRIOR:
#                 os.makedirs(
#                     f'plots_with_priorless_regualrization', exist_ok=True)
#             os.makedirs(
#                 f'plots_with_regualrization', exist_ok=True)
#             ax[0].axis('off')
#             ax[1].axis('off')
#             ax[0].imshow(true_map)
#             ax[0].set_title('True Map')
#             ax[1].imshow(pred_map)
#             ax[1].set_title('Predicted Map')
#             if ZERO_PRIOR:
#                 plt.savefig(
#                     f'plots_with_priorless_regualrization/map_{TYPE}_gen{t+1}.png')
#             plt.savefig(
#                 f'plots_with_regualrization/map_{TYPE}_gen{t+1}.png')
#     plt.close()
#     true_map = env.map_star
#     pred_map = theta_hat.reshape(SIZE, SIZE)
#     return true_map, pred_map
