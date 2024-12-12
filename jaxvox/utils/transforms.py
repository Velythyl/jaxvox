import functools

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R

# https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / jnp.linalg.norm(vec1)).reshape(3), (vec2 / jnp.linalg.norm(vec2)).reshape(3)
    v = jnp.cross(a, b)
    c = jnp.dot(a, b)
    s = jnp.linalg.norm(v)
    kmat = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = jnp.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def _transform_point(rot_mat, point):
    return rot_mat.dot(point)

def transform_points(rot_mat, points):
    points = jnp.atleast_2d(points)
    return jax.vmap(functools.partial(_transform_point, rot_mat=rot_mat))(point=points)
