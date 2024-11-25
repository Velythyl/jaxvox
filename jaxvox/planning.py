import functools

import jax.random

from jaxvox._jaxvox import VoxGrid

import jax.numpy as jnp

from jaxvox.samplers import inside_clipped_cone, base_cone
from split_key import split_key
from transforms import rotation_matrix_from_vectors, transform_points


def gen_waypoint_fromto(rng, p1, target, dist, dist_tol, radius_tol):
    mindist = (dist-dist_tol)
    maxdist =(dist+dist_tol)
    waypoint = inside_clipped_cone(rng, min_height_a=mindist, height_a=maxdist, base_radius_b=radius_tol)

    direction = target-p1
    direction = direction / jnp.linalg.norm(direction)
    rot_mat = rotation_matrix_from_vectors(jnp.array([0, 1, 0]), direction)
    hasnan = jnp.isnan(rot_mat).any().astype(jnp.float32)
    rot_mat = jnp.nan_to_num(rot_mat, nan=-9999, neginf=-9999, posinf=-9999)    # random bullshit, will not actually be used, just need to get rid of nans and infs

    waypoint = transform_points(rot_mat, waypoint) * (1-hasnan) + waypoint * hasnan
    waypoint = waypoint.squeeze()
    return waypoint + p1

#@functools.partial(jax.jit, static_argnums=(3,))


#@functools.partial(jax.jit, static_argnums=(3,))
def gen_path(key, start_point, end_point, num_waypoints, dist_tol=0.1, radius_tol=0.1):
    z_adjusted_endpoint = end_point.at[-1].set(start_point.squeeze()[-1])

    avg_dist = jnp.linalg.norm(end_point - start_point) / num_waypoints
    num_waypoints = num_waypoints - 1

    carry = (key, start_point, z_adjusted_endpoint, avg_dist, dist_tol, radius_tol)

    def scan_waypoints(carry, waypoint_id):
        (key, start_point, z_adjusted_endpoint, avg_dist, dist_tol, radius_tol) = carry
        key, rng = jax.random.split(key)

        waypoint = gen_waypoint_fromto(rng, start_point, z_adjusted_endpoint, avg_dist, dist_tol, radius_tol)
        carry = (key, waypoint, z_adjusted_endpoint, avg_dist, dist_tol, radius_tol)
        return carry, waypoint

    carry, waypoints = jax.lax.scan(scan_waypoints, carry, jnp.arange(num_waypoints))
    return jnp.vstack([waypoints, end_point[None]])

@functools.partial(jax.jit, static_argnums=(1,4))
def gen_paths(key, num_paths, start_point, end_point, num_waypoints, dist_tol=0.1, radius_tol=0.1):
    bound_gen_path = functools.partial(gen_path, start_point=start_point, end_point=end_point, num_waypoints=num_waypoints, dist_tol=dist_tol, radius_tol=radius_tol)
    _, num_key_keys = split_key(key, num_paths)
    return jax.vmap(bound_gen_path)(num_key_keys)

def path_to_pairs(paths):
    paths = jnp.atleast_3d(paths)
    def do_path(path):
        x = path
        y = jnp.concatenate([x[1:], x[-2][None]], axis=0)

        return jnp.concatenate([x[None], y[None]], axis=0)
    return jax.vmap(do_path)(paths)


if __name__ == "__main__":
    """
    positions_dict = {
        "alice_gripper": (0.29, 0.06, 0.51),
        "bob_gripper": (0.35, 1.05, 0.62),
        "bin_front_left": (0.25, 0.41, 0.36),
        "bin_front_right": (0.65, 0.41, 0.36),
        "bin_front_middle": (0.45, 0.41, 0.36),
        "bin_back_left": (0.25, 0.58, 0.36),
        "bin_back_right": (0.65, 0.58, 0.36),
        "bin_back_middle": (0.45, 0.58, 0.36),
        "apple": (-0.08, 0.58, 0.30),
        "banana": (-0.76, 0.39, 0.27),
        "milk": (-0.06, 0.39, 0.47),
        "soda_can": (-0.47, 0.40, 0.33),
        "bread": (-0.74, 0.59, 0.31),
        "cereal": (-0.22, 0.37, 0.44),
    }
    positions_dict = {k: jnp.array(v) for k, v in positions_dict.items()}
    """

    #  {'Alice': ('apple', 'apple_top', 1), 'Bob': ('banana', 'banana_top', 1)} inhand: {'Alice': None, 'Bob': None}
    #  {'Alice': 'PICK apple PATH [(0.29, 0.06, 0.51), (0.1, 0.22, 0.51), (-0.08, 0.4, 0.51), (-0.08, 0.58, 0.51)]',
    #  'Bob': 'PICK banana PATH [(0.35, 0.85, 0.62), (0.05, 0.62, 0.62), (-0.25, 0.50, 0.62), (-0.76, 0.39, 0.62)]'}

    import open3d as o3d
    import json

    pcd = o3d.io.read_point_cloud("./data/1/pcd.pcd")

    with open("./data/1/poses.json") as f:
        positions_dict = json.loads(f.read())
    positions_dict = {k: jnp.array(v) for k, v in positions_dict.items()}

    o3d_voxelgrid_from_point_cloud = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.02)
    voxgrid, attrmanager = VoxGrid.from_open3d(o3d_voxelgrid_from_point_cloud, import_attrs=True, return_attrmanager=True)
    attrmanager.set_default_value((255,0,0))

    for k, v in positions_dict.items():
        assert (v >= voxgrid.minbound).all()
        assert (v <= voxgrid.maxbound).all()

    key = jax.random.PRNGKey(0)
    test_waypoints = gen_paths(key, 10, jnp.array(positions_dict["alice"]), jnp.array(positions_dict["apple"]), 4, dist_tol=voxgrid.voxel_size*2, radius_tol=voxgrid.voxel_size*2)

    waypoint_pairs = path_to_pairs(test_waypoints)
    voxgrid = voxgrid.raycast(waypoint_pairs)

    #test_waypoints = test_waypoints.reshape(10*4,3)
    #test_voxels = voxgrid.point_to_voxel(test_waypoints)

    #test_voxels = test_voxels[0]
    #voxgrid = voxgrid.set_voxel(test_voxels)
    #test_values = voxgrid.is_voxel_set(test_voxels)
    #print(test_values)

    voxgrid.display_as_o3d(attrmanager)


