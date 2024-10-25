import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import *


class Solver:
    def __init__(self, max_iteration=1000) -> None:
        self.max_iteration = max_iteration

    def solve_tikhonov(self, A, b, lambda_):
        A, b = jnp.array(A), jnp.array(b)
        return jnp.linalg.pinv(A.T @ A + lambda_ * jnp.eye(A.shape[1])) @ A.T @ b, A.T @ A
    
    def solve_with_LOG_kernel(self, A, b, prior_value=0.1):
        A, b = jnp.array(A), jnp.array(b)
        theta_prior = jnp.ones(A.shape[1]) * prior_value
        log_matrix = LOG_kernel(A.shape[1])
        log_matrix = jnp.dot(log_matrix, log_matrix.T)
        term1 = jnp.linalg.pinv(A.T @ A + log_matrix)
        term2 = jnp.dot(A.T,b.reshape(-1,1)) + jnp.dot(log_matrix, theta_prior.reshape(-1,1))
        result = jnp.dot(term1, term2)
        
        print(theta_prior.shape,log_matrix.shape,term1.shape,term2.shape,result.shape)
        return result.reshape((-1,)), term1
