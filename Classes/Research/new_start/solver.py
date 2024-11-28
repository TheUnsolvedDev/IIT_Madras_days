import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils import *

# convolve_parallel = jax.vmap(lambda col,kernel: jnp.convolve(col, kernel, mode='same'), in_axes=1, out_axes=1)


def conv_parallel(x, kernel):
    convolve_parallel = jax.vmap(lambda col: jnp.convolve(col, kernel, mode='same'))
    return convolve_parallel(x)

class Solver:
    def __init__(self, max_iteration=1000) -> None:
        self.max_iteration = max_iteration

    def solve_tikhonov(self, A, b, lambda_):
        A, b = jnp.array(A), jnp.array(b)
        return jnp.linalg.pinv(A.T @ A + lambda_ * jnp.eye(A.shape[1])) @ A.T @ b, A.T @ A
    
    def solve_LOG_kernel(self, A, b, prior_value=0.1):
        A, b = jnp.array(A), jnp.array(b)
        theta_prior = jnp.ones(A.shape[1]) * prior_value
        new_A = conv_parallel(A, kernel=jnp.array([1,-2,1]))
        term1 = jnp.linalg.pinv(new_A.T @ new_A) + theta_prior
        term2 = jnp.dot(A.T,b.reshape(-1,1)) 
        result = jnp.dot(term1, term2)
        return result.reshape((-1,)), term1

if __name__ == "__main__":
    
    n,d,k = 2,5,3
    A = jax.random.normal(jax.random.PRNGKey(0), (n,d))
    b = jax.random.normal(jax.random.PRNGKey(1), (k,))
    print(conv_parallel(A,b) == jnp.array([jnp.convolve(i,b,mode='same') for i in A]))