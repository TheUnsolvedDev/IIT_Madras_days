import numpy as np
import jax
import jax.numpy as jnp


class History:
    def __init__(self, size=5, lambda_=0.01):
        self.size = size
        self.lambda_ = lambda_
        self.reset()

    def reset(self):
        self.V_t = jnp.eye(self.size**2)
        self.uncertainity, self.A, self.r = [], [], []

    def compute_uncertainity_and_reconstruction(self):
        if len(self.A) == 0:
            uncertainity = jnp.linalg.eigh(self.V_t)[0]
            reconstruction = jnp.zeros(self.size**2)
        else:
            self.V_t = jnp.eye(self.size**2)*self.lambda_ + \
                jnp.array(self.A).T @ jnp.array(self.A)
            uncertainity = jnp.linalg.eigh(self.V_t)[0]
            reconstruction = jnp.linalg.inv(
                self.V_t) @ jnp.array(self.A).T @ jnp.array(self.r)
        return uncertainity.reshape(self.size,self.size), reconstruction.reshape(self.size,self.size)

    def compute_theta_t(self):
        pass

    def add(self, A_t, r_t):
        A_t = jnp.array(A_t).reshape(-1)
        self.A.append(A_t)
        self.r.append(r_t)
        # uncertainity =


if __name__ == "__main__":
    obj = History()
