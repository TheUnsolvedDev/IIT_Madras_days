import numpy as np
import jax.numpy as jnp
from scipy.optimize import nnls
import pylops
import jax
import functools
from utils import *


class StrategiesReconstruction:

    L = None
    H = None
    Theta_prior = None

    @staticmethod
    # with regularization and lambda identity matrix
    def with_tikhonov_and_lambda_identity(A_ts, B_ts, **kwargs):
        """
        with regularization and lambda identity matrix

        Args:
            A_ts (array): Input array A_ts.
            B_ts (array): Input array B_ts.
            **kwargs: Additional keyword arguments.
                lamda (float): The regularization parameter.

        Returns:
            tuple: A tuple containing the reconstructed theta and a tuple of identity matrix and lamda.
                theta (ndarray): The reconstructed theta.
                (ndarray, float): Tuple of identity matrix and lamda.
        """
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = jnp.eye(kwargs['size']**2)
            StrategiesReconstruction.H = kwargs['lamda']*jnp.dot(
                StrategiesReconstruction.L, StrategiesReconstruction.L)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                jnp.zeros((kwargs['size']**2, 1))

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = StrategiesReconstruction.H + V_t
        if kwargs['lamda'] == 0:
            theta_hat_1 = jnp.linalg.pinv(M_t)
        else:
            theta_hat_1 = jnp.linalg.inv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + StrategiesReconstruction.H@StrategiesReconstruction.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat, (StrategiesReconstruction.H, kwargs['lamda'])

    @staticmethod
    # LOG kernel reconstruction with sparsity prior
    def with_Gauss_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        """
        Gauss kernel reconstruction with sparsity prior.

        Args:
            A_ts (list): A list of time series data.
            B_ts (list): A list of time series data.
            **kwargs: Additional keyword arguments.
                size (int): The size of the kernel.
                lamda (float): The regularization parameter.
                sparsity (float): The sparsity parameter.

        Returns:
            tuple: A tuple containing the reconstructed theta and a tuple of H and lamda.
                theta (ndarray): The reconstructed theta.
                H (ndarray): The kernel matrix.
                lamda (float): The regularization parameter.

        """
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = Gauss_kernel(kwargs['size'])
            StrategiesReconstruction.H = kwargs['lamda']*jnp.dot(
                StrategiesReconstruction.L, StrategiesReconstruction.L)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                jnp.zeros((kwargs['size']**2, 1))

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = jnp.linalg.inv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + StrategiesReconstruction.H@StrategiesReconstruction.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat, (StrategiesReconstruction.H, kwargs['lamda'])

    @staticmethod
    # LOG kernel reconstruction with sparsity prior
    def with_LOG_regularization_and_sparsity_prior(A_ts: list, B_ts: list, **kwargs: dict) -> tuple:
        """
        LOG kernel reconstruction with sparsity prior.

        Args:
            A_ts (list): A list of time series data.
            B_ts (list): A list of time series data.
            **kwargs (dict): Additional keyword arguments.
                size (int): The size of the kernel.
                lamda (float): The regularization parameter.
                sparsity (float): The sparsity parameter.

        Returns:
            tuple: A tuple containing the reconstructed theta and a tuple of H and lamda.
                theta (ndarray): The reconstructed theta.
                H (ndarray): The kernel matrix.
                lamda (float): The regularization parameter.

        """
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = LOG_kernel(kwargs['size'])
            StrategiesReconstruction.H = kwargs['lamda']*jnp.dot(
                StrategiesReconstruction.L, StrategiesReconstruction.L)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                jnp.zeros((kwargs['size']**2, 1))

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = jnp.linalg.inv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + StrategiesReconstruction.H@StrategiesReconstruction.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat, (StrategiesReconstruction.H, kwargs['lamda'])
    
    @staticmethod
    # Sharpen kernel reconstruction with sparsity prior
    def with_Sharpen_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        """
        Sharpenkernel reconstruction with sparsity prior.

        Args:
            A_ts (list): A list of time series data.
            B_ts (list): A list of time series data.
            **kwargs: Additional keyword arguments.
                size (int): The size of the kernel.
                lamda (float): The regularization parameter.
                sparsity (float): The sparsity parameter.

        Returns:
            tuple: A tuple containing the reconstructed theta and a tuple of H and lamda.
                theta (ndarray): The reconstructed theta.
                H (ndarray): The kernel matrix.
                lamda (float): The regularization parameter.

        """
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = Sharpen_kernel(kwargs['size'])
            StrategiesReconstruction.H = kwargs['lamda']*jnp.dot(
                StrategiesReconstruction.L, StrategiesReconstruction.L)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                jnp.zeros((kwargs['size']**2, 1))

        V_t = jnp.dot(jnp.array(A_ts).T, jnp.array(A_ts))
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = jnp.linalg.inv(M_t)
        theta_hat_2 = (jnp.array(A_ts).T @
                       jnp.array(B_ts).reshape((-1, 1)) + StrategiesReconstruction.H@StrategiesReconstruction.Theta_prior)
        theta_hat = jax.device_get(jnp.dot(theta_hat_1, theta_hat_2))
        return theta_hat, (StrategiesReconstruction.H, kwargs['lamda'])


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
        return theta_hat, (StrategiesReconstruction.H, 1)

    @staticmethod
    # Non negative least square regularization
    def with_non_negative_least_square(A_ts, B_ts, **kwargs):
        projection, received = np.array(A_ts), np.array(B_ts).reshape((-1,))
        theta_hat = nnls(projection, received)[0].reshape(-1, 1)
        return theta_hat


class StartegiesAction:
    def __init__(self, all_possible_actions) -> None:
        self.all_possible_actions = all_possible_actions

    @functools.partial(jax.jit, static_argnums=(0,))
    def min_eigenvalue_info_action(self, theta_hat):
        eig_vals, eig_vecs = jnp.linalg.eigh(theta_hat)
        sorted_indices = jnp.argsort(eig_vals)
        sorted_eig_vals = eig_vals[sorted_indices]
        sorted_eig_vecs = eig_vecs[sorted_indices]
        return jnp.argmin(jnp.dot(self.all_possible_actions, sorted_eig_vecs[0]))

    def random_action(self, theta_hat):
        return np.random.randint(len(self.all_possible_actions))
