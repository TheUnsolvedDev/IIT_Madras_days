import tensorflow as tf
import pylops
import numpy as np

from utils import *


class StrategiesReconstruction:

    L = None
    H = None
    Theta_prior = None

    @staticmethod
    def with_tikhonov_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        A_ts,B_ts = np.array(A_ts),np.array(B_ts)
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = np.eye(kwargs['size']**2)
            StrategiesReconstruction.H = tf.cast(kwargs['lamda']*tf.matmul(
                StrategiesReconstruction.L, StrategiesReconstruction.L),tf.float32)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                np.ones((kwargs['size']**2, 1))

        V_t = tf.matmul(tf.transpose(A_ts), A_ts)
        M_t = StrategiesReconstruction.H + V_t
        if kwargs['lamda'] == 0:
            theta_hat_1 = tf.linalg.pinv(M_t)
        else:
            theta_hat_1 = tf.linalg.inv(M_t)
        theta_hat = tf.matmul(theta_hat_1, tf.matmul(
            tf.transpose(A_ts), B_ts.reshape(-1, 1)))
        theta_hat += tf.matmul(StrategiesReconstruction.H,
                               StrategiesReconstruction.Theta_prior)
        return theta_hat.numpy(), (StrategiesReconstruction.L, kwargs['lamda'])

    @staticmethod
    def with_Gauss_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        A_ts,B_ts = np.array(A_ts),np.array(B_ts)
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = Gauss_kernel(kwargs['size'])
            StrategiesReconstruction.H = tf.cast(kwargs['lamda']*tf.matmul(
                StrategiesReconstruction.L, StrategiesReconstruction.L),tf.float32)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                np.ones((kwargs['size']**2, 1), dtype=np.float32)

        V_t = tf.matmul(tf.transpose(A_ts), A_ts)
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = tf.linalg.inv(M_t)
        theta_hat = tf.matmul(theta_hat_1, tf.matmul(
            tf.transpose(A_ts), B_ts.reshape(-1, 1)))
        theta_hat += StrategiesReconstruction.H @ StrategiesReconstruction.Theta_prior
        return theta_hat.numpy(), (StrategiesReconstruction.L, kwargs['lamda'])

    @staticmethod
    def with_Sharpen_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        A_ts,B_ts = np.array(A_ts),np.array(B_ts)
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = Sharpen_kernel(kwargs['size'])
            StrategiesReconstruction.H = tf.cast(kwargs['lamda']*tf.matmul(
                StrategiesReconstruction.L, StrategiesReconstruction.L),tf.float32)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                np.ones((kwargs['size']**2, 1))

        V_t = tf.matmul(tf.transpose(A_ts), A_ts)
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = tf.linalg.inv(M_t)
        theta_hat = tf.matmul(theta_hat_1, tf.matmul(
            tf.transpose(A_ts), B_ts.reshape(-1, 1)))
        theta_hat += StrategiesReconstruction.H @ StrategiesReconstruction.Theta_prior
        return theta_hat.numpy(), (StrategiesReconstruction.L, kwargs['lamda'])

    @staticmethod
    def with_LOG_regularization_and_sparsity_prior(A_ts, B_ts, **kwargs):
        A_ts,B_ts = np.array(A_ts),np.array(B_ts)
        if StrategiesReconstruction.L is None:
            StrategiesReconstruction.L = LOG_kernel(kwargs['size'])
            StrategiesReconstruction.H = tf.cast(kwargs['lamda']*tf.matmul(
                StrategiesReconstruction.L, StrategiesReconstruction.L),tf.float32)
            StrategiesReconstruction.Theta_prior = kwargs['sparsity'] * \
                np.ones((kwargs['size']**2, 1))

        V_t = tf.matmul(tf.transpose(A_ts), A_ts)
        M_t = StrategiesReconstruction.H + V_t
        theta_hat_1 = tf.linalg.inv(M_t)
        theta_hat = tf.matmul(theta_hat_1, tf.matmul(
            tf.transpose(A_ts), B_ts.reshape(-1, 1)))
        theta_hat += StrategiesReconstruction.H @ StrategiesReconstruction.Theta_prior
        return theta_hat.numpy(), (StrategiesReconstruction.L, kwargs['lamda'])

    @staticmethod
    def with_laplacian_regularization(A_ts, B_ts, **kwargs):
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

class StartegiesAction:
    def __init__(self, all_possible_actions) -> None:
        self.all_possible_actions = all_possible_actions
    
    def min_eigenvalue_info_action(self,theta_hat):
        eig_vals,eig_vecs = tf.linalg.eigh(theta_hat)
        return np.argmin(tf.matmul(self.all_possible_actions,tf.reshape(eig_vecs[0],(-1,1))))

    def random_action(self, theta_hat):
        return np.random.randint(len(self.all_possible_actions))
