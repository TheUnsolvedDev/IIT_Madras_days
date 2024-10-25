import jax
import flax

def show_summary(Q, x, rng = jax.random.PRNGKey(0)):
    print(flax.linen.tabulate(Q, rng, compute_flops=True, compute_vjp_flops=True)(*x))