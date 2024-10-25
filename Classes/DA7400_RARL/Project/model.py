import jax.numpy as jnp
import jax
import numpy as np
import flax
from typing import *

from model_utils import *

class ActorCriticModel(flax.linen.Module):
    action_dim: int
    activation: str = 'gelu'

    @flax.linen.compact
    def __call__(self, x):
        if self.activation == 'relu':
            activation = flax.linen.relu
        elif self.activation == 'gelu':
            activation = flax.linen.gelu
        else:
            activation = flax.linen.tanh

        actor_mean = flax.linen.Dense(64, kernel_init=jax.nn.initializers.orthogonal(np.sqrt(2)
        ), bias_init=jax.nn.initializers.normal(stddev=1e-2))(x)
        actor_mean = activation(actor_mean)
        actor_mean = flax.linen.Dense(64, kernel_init=jax.nn.initializers.orthogonal(np.sqrt(2)
        ), bias_init=jax.nn.initializers.normal(stddev=1e-2))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = flax.linen.Dense(self.action_dim, kernel_init=jax.nn.initializers.orthogonal(np.sqrt(2)
        ), bias_init=jax.nn.initializers.normal(stddev=1e-2))(actor_mean)
        pi = jax.nn.softmax(actor_mean, axis=-1)

        critic = flax.linen.Dense(64, kernel_init=jax.nn.initializers.orthogonal(np.sqrt(2)
        ), bias_init=jax.nn.initializers.normal(stddev=1e-2))(x)
        critic = activation(critic)
        critic = flax.linen.Dense(64, kernel_init=jax.nn.initializers.orthogonal(np.sqrt(2)
        ), bias_init=jax.nn.initializers.normal(stddev=1e-2))(critic)
        critic = activation(critic)
        critic = flax.linen.Dense(1, kernel_init=jax.nn.initializers.orthogonal(np.sqrt(2)
        ), bias_init=jax.nn.initializers.normal(stddev=1e-2))(critic)

        return pi,jnp.squeeze(critic, axis=-1)
    
if __name__ == '__main__':
    model = ActorCriticModel(action_dim=3)
    print(show_summary(model, [jnp.ones((1, 64))]))
    
