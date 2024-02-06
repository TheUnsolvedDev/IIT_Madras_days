import numpy as np
import jax.numpy as jnp
from scipy.optimize import nnls
import pylops
import jax
from utils import *


class Strategies:

    L = None
    H = None
    Theta_prior = None

    @staticmethod
    def without_regularization(A_ts, B_ts, **kwargs):
        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        theta_hat_1 = jnp.linalg.pinv(V_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)))
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    def with_regularization(A_ts, B_ts, **kwargs):
        if Strategies.L is None:
            Strategies.L = LOG_kernel(kwargs['size'])
            Strategies.H = jnp.dot(Strategies.L, Strategies.L)
            Strategies.Theta_prior = kwargs['sparsity'] * \
                jnp.zeros((kwargs['size']**2, 1))

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = Strategies.H + V_t
        theta_hat_1 = jnp.linalg.pinv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + Strategies.H@Strategies.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    def with_regularization_and_zero_prior(A_ts, B_ts, **kwargs):
        if Strategies.L is None:
            Strategies.L = LOG_kernel(kwargs['size'])
            Strategies.H = jnp.dot(Strategies.L, Strategies.L)
            Strategies.Theta_prior = 0

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = Strategies.H + V_t
        theta_hat_1 = jnp.linalg.pinv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + Strategies.H@Strategies.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    def with_tikhonov_regularization(A_ts, B_ts, **kwargs):
        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        theta_hat = pylops.optimization.leastsquares.regularized_inversion(
            projection,
            received,
            [pylops.Laplacian(dims=(kwargs['size'], kwargs['size']), edge=True,
                              weights=(3, 3), dtype="float32")],
            epsRs=[np.sqrt(0.1)],
            **dict(damp=np.sqrt(1e-4), iter_lim=50, show=0)
        )[0].reshape(-1, 1)
        return theta_hat

    @staticmethod
    def with_non_negative_least_square(A_ts, B_ts, **kwargs):
        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        theta_hat = nnls(projection, received)[0].reshape(-1, 1)
        return theta_hat

    @staticmethod
    def with_tikhonov_and_nnls(A_ts, B_ts, **kwargs):
        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        num_variables = projection.shape[1]
        PROJ_TKNV = np.concatenate([projection,
                                    np.sqrt(0.1) *
                                    np.eye(num_variables)
                                    ])
        RECV_TKNV = np.concatenate(
            [received, np.zeros(num_variables)])
        theta_hat = nnls(PROJ_TKNV, RECV_TKNV)[0].reshape((-1, 1))
        return theta_hat
