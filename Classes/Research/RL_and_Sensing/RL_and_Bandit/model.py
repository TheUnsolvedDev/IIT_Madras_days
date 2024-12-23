import flax.linen
import flax.linen
import jax
import jax.numpy as jnp
import numpy as np
import flax

class QModel(flax.linen.Module):
    num_actions: int
    
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Conv(features=32, kernel_size=(8, 8), padding='SAME')(x)
        x = flax.linen.pooling.avg_pool(x, window_shape=(2, 2))
        x = jax.nn.relu(x)
        x = flax.linen.Conv(features=64, kernel_size=(4, 4), padding='SAME')(x)
        x = flax.linen.pooling.avg_pool(x, window_shape=(2, 2))
        x = jax.nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = flax.linen.Dense(features=512)(x)
        x = jax.nn.relu(x)
        x = flax.linen.Dense(features=self.num_actions)(x)
        return x
        

if __name__ == "__main__":
    model = QModel(num_actions=4)
    print(model)
    
    x = jnp.ones((1, 20, 20, 4))
    print(model.tabulate(jax.random.PRNGKey(0), x))