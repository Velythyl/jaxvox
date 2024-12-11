import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnums=(1,))
def split_key(key, num_keys):
    key, *rng = jax.random.split(key, num_keys + 1)
    rng = jnp.reshape(jnp.stack(rng), (num_keys, 2))
    return key, rng