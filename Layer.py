import equinox as eqx
import jax
import jax.numpy as jnp

class MembershipFunctions(eqx.Module):
    centers: jax.Array
    widths: jax.Array

    def __init__(self, num_inputs: int, num_mfs: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, 2)
        self.centers = jax.random.uniform(keys[0], (num_inputs, num_mfs))
        self.widths = jax.random.uniform(keys[1], (num_inputs, num_mfs)) + 0.1  # ensure positive

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.exp(-((x[:, None] - self.centers) ** 2) / (2 * self.widths ** 2))