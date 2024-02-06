# import jax
# print(jax.devices())

# import torch
# print(torch.cuda.is_available())

# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))

import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm

from environment import *
from utils import *

SIZE = 8
SPARSITY = 0.75


def main():
    env = Environment(sparsity=SPARSITY, size=SIZE)
    env.reset()

    A_ts, B_ts, T = [], [], 3000
    L = LOG_kernel(SIZE)
    H = jnp.dot(L, L)
    action = np.random.choice(SIZE**4)
    delta = 0.1
    theta_prior = SPARSITY*jnp.ones((SIZE*SIZE, 1))

    all_possible_actions = np.array(
        [env._action_vector(i) for i in range(env.total_actions)])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for t in tqdm.tqdm(range(T)):
        A_t, B_t = env.step(action)
        A_ts.append(A_t)
        B_ts.append(B_t)
        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))

        M_t = H + V_t
        theta_hat_1 = jnp.linalg.pinv(V_t+H)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + H@theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))

        # print(H)

        C_t = jnp.linalg.eigh(L)[0][-1] + jnp.sqrt(2*np.log(1/delta) +
                                                   np.log(jnp.linalg.det(M_t)/(jnp.linalg.det(H))))
        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        action = jnp.argmax(jnp.dot(all_possible_actions, eig_vecs[0]))
        print(convert_actions(action, SIZE), B_t)

        if (t+1) % 100 == 0:
            true_map = env.map_star
            pred_map = theta_hat.reshape(SIZE, SIZE)
            # img = np.hstack([true_map, pred_map])
            ax[0].imshow(true_map)
            ax[0].set_title('True Map')
            ax[1].imshow(pred_map)
            ax[1].set_title('Predicted Map')
            plt.savefig(f'plots3/map_gen{t+1}.png')


if __name__ == '__main__':
    main()
