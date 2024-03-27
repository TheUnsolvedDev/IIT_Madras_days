import numpy as np
import jax.numpy as jnp
from scipy.optimize import nnls
import pylops
import jax
from utils import *


class StrategiesReconstruction:

    L = None
    H = None
    Theta_prior = None

    @staticmethod
    # without any regularization
    def without_regularization(A_ts, B_ts, **kwargs):
        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        theta_hat_1 = jnp.linalg.pinv(V_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)))
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    # with regularization and identity matrix
    def with_tikhonov_and_identity(A_ts, B_ts, **kwargs):
        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        theta_hat_1 = jnp.linalg.pinv(V_t + jnp.eye(V_t.shape[0]))
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)))
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    # with regularization and lambda identity matrix
    def with_tikhonov_and_lambda_identity(A_ts, B_ts, **kwargs):
        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        theta_hat_1 = jnp.linalg.pinv(
            V_t + kwargs['lamda']*jnp.eye(V_t.shape[0]))
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)))
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    # LOG kernel reconstruction with sparsity prior
    def with_LOG_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = LOG_kernel(kwargs['size'])
            StrategiesReconstruction.H = jnp.dot(
                StrategiesReconstruction.L, StrategiesReconstruction.L)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                jnp.zeros((kwargs['size']**2, 1))

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = jnp.linalg.pinv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + StrategiesReconstruction.H@StrategiesReconstruction.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    # LOG kernel reconstruction with zero prior
    def with_LOG_regularization_and_zero_prior(A_ts, B_ts, **kwargs):
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = LOG_kernel(kwargs['size'])
            StrategiesReconstruction.H = jnp.dot(
                StrategiesReconstruction.L, StrategiesReconstruction.L)
            StrategiesReconstruction.Theta_prior = 0

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = jnp.linalg.pinv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + StrategiesReconstruction.H@StrategiesReconstruction.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat

    @staticmethod
    # Laplacian Regularization Ayon sir's
    def with_Laplacian_regularization(A_ts, B_ts, **kwargs):
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
    # Non negative least square regularization
    def with_non_negative_least_square(A_ts, B_ts, **kwargs):
        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        theta_hat = nnls(projection, received)[0].reshape(-1, 1)
        return theta_hat


class StartegiesAction:
    def __init__(self, all_possible_actions) -> None:
        self.all_possible_actions = all_possible_actions

    def min_eigenvalue_info_action(self, theta_hat):
        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        sorted_indices = jnp.argsort(eig_vals)
        sorted_eig_vals = eig_vals[sorted_indices]
        sorted_eig_vecs = eig_vecs[sorted_indices]
        return jnp.argmin(jnp.dot(self.all_possible_actions, sorted_eig_vecs[0]))

    def random_action(self, theta_hat):
        return np.random.randint(len(self.all_possible_actions))
