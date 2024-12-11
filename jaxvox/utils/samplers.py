import functools

import jax.random
import jax.numpy as jnp

from transforms import rotation_matrix_from_vectors


def rand(key, minval=0, maxval=1):
    return jax.random.uniform(key, shape=(1,), minval=minval, maxval=maxval)

def inside_cone(key, height_a, base_radius_b):
    #https://stackoverflow.com/questions/41749411/uniform-sampling-by-volume-within-a-cone
    return inside_clipped_cone(key, 4, height_a, base_radius_b)


def inside_clipped_cone(key, min_height_a, height_a, base_radius_b):
    # https://stackoverflow.com/questions/41749411/uniform-sampling-by-volume-within-a-cone
    key, rng = jax.random.split(key)
    h_rand_component = rand(rng, minval=min_height_a/height_a, maxval=1)    # 0 is 0, 1 is height_a
    h = height_a * (h_rand_component) ** (1 / 3)

    key, rng = jax.random.split(key)
    r = (base_radius_b / height_a) * h * jnp.sqrt(rand(rng))

    key, rng = jax.random.split(key)
    t = 2 * jnp.pi * rand(rng)

    x = r * jnp.cos(t)
    y = h
    z = r * jnp.sin(t)

    return jnp.hstack([x, y, z])

def side_cone(key, height_a, base_radius_b):
    key, rng = jax.random.split(key)
    h = height_a * jnp.sqrt(rand(rng))
    r = (base_radius_b / height_a) * h
    t = 2 * jnp.pi * rand(key)

    x = r * jnp.cos(t)
    y = h
    z = r * jnp.sin(t)

    return jnp.hstack([x, y, z])

def base_cone(key, height_a, base_radius_b):
    h = height_a

    key, rng = jax.random.split(key)
    r = base_radius_b * jnp.sqrt(rand(rng))

    key, rng = jax.random.split(key)
    t = 2 * jnp.pi * rand(rng)

    x = r * jnp.cos(t)
    y = jnp.atleast_1d(h)
    z = r * jnp.sin(t)

    return jnp.hstack([x, y, z])

if __name__ == "__main__":
    from jaxvox.jaxutils.split_key import split_key
    _, keys = split_key(jax.random.PRNGKey(0), 1000)

    points = jax.vmap(functools.partial(side_cone, height_a=5, base_radius_b=2))(keys)

    means = jnp.array([jnp.mean(points[:,0], axis=0), jnp.mean(points[:,1], axis=0), jnp.mean(points[:,2], axis=0)])


    rot_mat = rotation_matrix_from_vectors(jnp.array([0,1,0]), jnp.array([0,0,1]))
    #points = transform_points(rot_mat, points)

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("TkAgg")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()