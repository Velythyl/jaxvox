import functools

import jax.random


import jax.numpy as jnp

from jaxvox._jaxvox._jaxvox import VoxGrid
from jaxvox.jaxutils.split_key import split_key
from jaxvox.utils.samplers import inside_clipped_cone
from jaxvox.utils.transforms import rotation_matrix_from_vectors, transform_points


def gen_waypoint_fromto(rng, p1, target, dist, dist_tol, radius_tol):
    mindist = (dist - dist_tol)
    maxdist = (dist + dist_tol)
    waypoint = inside_clipped_cone(rng, min_height_a=mindist, height_a=maxdist, base_radius_b=radius_tol)

    direction = target - p1
    direction = direction / jnp.linalg.norm(direction)
    rot_mat = rotation_matrix_from_vectors(jnp.array([0, 1, 0]), direction)
    hasnan = jnp.isnan(rot_mat).any().astype(jnp.float32)
    rot_mat = jnp.nan_to_num(rot_mat, nan=-9999, neginf=-9999,
                             posinf=-9999)  # random bullshit, will not actually be used, just need to get rid of nans and infs

    waypoint = transform_points(rot_mat, waypoint) * (1 - hasnan) + waypoint * hasnan
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


@functools.partial(jax.jit, static_argnums=(1, 4))
def gen_paths(key, num_paths, start_point, end_point, num_waypoints, dist_tol=0.1, radius_tol=0.1):
    bound_gen_path = functools.partial(gen_path, start_point=start_point, end_point=end_point,
                                       num_waypoints=num_waypoints, dist_tol=dist_tol, radius_tol=radius_tol)
    _, num_key_keys = split_key(key, num_paths)
    return jax.vmap(bound_gen_path)(num_key_keys)


def path_to_pairs(paths):
    #paths = jnp.atleast_3d(paths)
    if len(paths.shape) == 2:
        paths = paths[None]

    def do_path(path):
        x = path
        y = jnp.concatenate([x[1:], x[-2][None]], axis=0)

        return jnp.concatenate([x[None], y[None]], axis=0)

    return jax.vmap(do_path)(paths)


def path_and_grid_have_collision(voxgrid, expand_size, genned_path):
    #waypoint_pairs = path_to_pairs(genned_path)
    #voxgrid2 = voxgrid.empty()
    #voxgrid2 = voxgrid2.raycast(waypoint_pairs, attrs=attrs)

    voxgrid2 = raypaths(voxgrid.empty(), genned_path[:-1], expand_size)

    return voxgrid.has_collision(voxgrid2) #((voxgrid2.grid > 1) + (voxgrid.grid > 1)) > 1).any()


@functools.partial(jax.jit, static_argnums=(2,3))
def raypaths(voxgrid_1, path, distance=0, attrs=None):
    if attrs is None:
        attrs = 1
    pairs_1 = path_to_pairs(path)
    voxgrid_1, (ray_points, ray_voxels, ray_voxels_attrs) = voxgrid_1.raycast(pairs_1, attrs=attrs, return_aux=True)

    if distance == 0:
        return voxgrid_1

    return voxgrid_1.set_voxel_neighbours(ray_voxels.reshape(-1, 3), distance=distance, include_corners=True,
                                          attrs=attrs)


def path_and_path_have_collision(voxgrid, path_1, expand_size, path_2):
    voxgrid_1 = raypaths(voxgrid.empty(), path_1, distance=expand_size, attrs=1)
    #pairs_1 = path_to_pairs(path_1)
    #voxgrid_1 = voxgrid.empty()
    #voxgrid_1, (ray_points, ray_voxels, ray_voxels_attrs) = voxgrid_1.raycast(pairs_1, attrs=1, return_aux=True)
    #voxgrid_1 = voxgrid_1.set_voxel_neighbours(ray_voxels.reshape(-1,3), distance=2, include_corners=True)

    #voxgrid_1 = voxgrid_1.dilate(1).dilate(1)

    #pairs_2 = path_to_pairs(path_2)
    #voxgrid_2 = voxgrid.empty()
    #voxgrid_2, (ray_points, ray_voxels, ray_voxels_attrs) = voxgrid_2.raycast(pairs_2, attrs=1, return_aux=True)
    #voxgrid_2 = voxgrid_2.dilate(1).dilate(1)
    #voxgrid_2 = voxgrid_2.set_voxel_neighbours(ray_voxels.reshape(-1,3), distance=2, include_corners=True)
    voxgrid_2 = raypaths(voxgrid.empty(), path_2, distance=expand_size, attrs=2)

    return voxgrid_1.has_collision(voxgrid_2) #  (voxgrid_1.attr_to_1().grid + voxgrid_2.attr_to_1().grid >= 2).sum() > 0

@functools.partial(jax.jit, static_argnums=(3,6))
def jitbatch_plan_single_robot(voxgrid, robot_position, target_position, batch_size=100, dist_tol=None, radius_tol=None, expand_size=2):
    if dist_tol is None:
        dist_tol = 2
    dist_tol = voxgrid.voxel_size * dist_tol
    if radius_tol is None:
        radius_tol = 10
    radius_tol = voxgrid.voxel_size * radius_tol

    key = jax.random.PRNGKey(0)
    key, rng = jax.random.split(key)
    genned_paths = gen_paths(rng, batch_size, jnp.array(robot_position), jnp.array(target_position), 4,
                             dist_tol=dist_tol, radius_tol=radius_tol)

    cloud_collision_mask = jax.vmap(functools.partial(path_and_grid_have_collision, voxgrid, expand_size))(genned_paths)
    return genned_paths, cloud_collision_mask




def plan_many_robots(voxgrid, robot_positions, target_positions, batch_size=100, dist_tol=None, radius_tol=None, expand_size=2):
    #@functools.partial(jax.jit, static_argnums=(3,))
    def _one_pass(voxgrid, robot_positions, target_positions, batch_size, dist_tol, radius_tol):
        all_genned_paths, object_collision_mask = jax.vmap(
            functools.partial(jitbatch_plan_single_robot, voxgrid=voxgrid, batch_size=batch_size, dist_tol=dist_tol, radius_tol=radius_tol, expand_size=expand_size))(robot_position=robot_positions,
                                                                                                     target_position=target_positions)
        paths = all_genned_paths
        #no_object_collision_mask = jnp.logical_not(object_collision_mask)
        #paths = []
        #for rnum in range(len(robot_positions)):
        #    paths.append(all_genned_paths[rnum][no_object_collision_mask[rnum]])

        #for pa in paths:
        #    if len(pa) == 0:
        #        return jnp.array([jnp.nan])
        if len(robot_positions) > 2:
            raise NotImplementedError()
        elif len(robot_positions) == 2:
            r1_paths = paths[0] #jnp.concatenate([robot_positions[0][None], ], axis=0)
            r2_paths = paths[1] #jnp.concatenate([robot_positions[1][None], all_genned_paths[1]], axis=0)

            def prepend_start_pos(rpath, start_pos):
                def for_each_path(start_pos, _rpath):
                    return jnp.concatenate([start_pos[None], _rpath], axis=0)
                return jax.vmap(functools.partial(for_each_path, start_pos))(rpath)
            #expanded_r1_start = jnp.repeat(r)

            r1_paths = prepend_start_pos(r1_paths, robot_positions[0])
            r2_paths = prepend_start_pos(r2_paths, robot_positions[1])

            expanded_r2_paths = jnp.repeat(r2_paths[None], batch_size, axis=0)

            def for_each_r1_path(voxgrid, r1_path, r2_paths, expand_size):
                def for_each_r2_path(voxgrid, r1_path, expand_size, r2_path):
                    return path_and_path_have_collision(voxgrid, r1_path, expand_size, r2_path)

                return jax.vmap(functools.partial(for_each_r2_path, voxgrid, r1_path, expand_size))(r2_paths)

            robot_robot_collision_mask = jax.vmap(functools.partial(for_each_r1_path, voxgrid=voxgrid, expand_size=expand_size))(r1_path=r1_paths,
                                                                                      r2_paths=expanded_r2_paths)
            r0_invalid = jnp.repeat(object_collision_mask[0][None], batch_size, axis=0)
            r1_invalid = jnp.repeat(object_collision_mask[1][:,None], batch_size, axis=1)
            any_collision_mask = robot_robot_collision_mask + r0_invalid + r1_invalid

            is_valid_plan = jnp.logical_not(any_collision_mask)
            nonzero = jnp.nonzero(is_valid_plan, size=1, fill_value=jnp.max(jnp.array(voxgrid.padded_grid_shape) * 2))
            #nonzero = jnp.array(nonzero, dtype=jnp.uint32).squeeze()
            return jnp.concatenate([
                all_genned_paths.at[0, nonzero[0]].get(mode="fill", fill_value=jnp.nan)[None],
                all_genned_paths.at[1, nonzero[1]].get(mode="fill", fill_value=jnp.nan)[None]]).squeeze()

    for pass_id in range(10):
        result = _one_pass(voxgrid, robot_positions, target_positions, batch_size, dist_tol, radius_tol)
        if jnp.isnan(result).any():
            result = None
            dist_tol += 1
            radius_tol += 1
        else:
            break

    if result is None:
        raise Exception("No valid plan for request.")

    return result


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
    voxgrid, attrmanager = VoxGrid.from_open3d(o3d_voxelgrid_from_point_cloud, import_attrs=True,
                                               return_attrmanager=True)
    attrmanager.set_default_value((255, 0, 0))


    #voxgrid.display_as_o3d(attrmanager)

    for k, v in positions_dict.items():
        assert (v >= voxgrid.minbound).all()
        assert (v <= voxgrid.maxbound).all()

    #voxgrid.display_as_o3d(attrmanager)

    #voxgrid = voxgrid.empty()

    paths = plan_many_robots(voxgrid, jnp.array([positions_dict["alice"], positions_dict["bob"]]),
                             jnp.array([positions_dict["apple"], positions_dict["cereal"]]), batch_size=10, dist_tol=2, radius_tol=6, expand_size=2)

    voxgrid = raypaths(voxgrid, paths[0], 3, 1)
    voxgrid = raypaths(voxgrid, paths[1], 3, 20)

    voxgrid.display_as_o3d(attrmanager)

    exit()
    key = jax.random.PRNGKey(0)
    test_waypoints = gen_paths(key, 10, jnp.array(positions_dict["alice"]), jnp.array(positions_dict["apple"]), 4,
                               dist_tol=voxgrid.voxel_size * 2, radius_tol=voxgrid.voxel_size * 5)

    waypoint_pairs = path_to_pairs(test_waypoints)
    attrs = jax.random.randint(key, (4,), minval=2, maxval=100) + 10

    voxgrid2 = voxgrid.empty()
    voxgrid2 = voxgrid2.raycast(waypoint_pairs[0], attrs=attrs)
    voxgrid2 = voxgrid2.dilate(1).dilate(1)

    #voxgrid2 = voxgrid.empty()
    #neighbours_voxels = voxgrid2.voxel_to_neighbours(jnp.array([[10, 10, 10], [20, 20, 20]]), 1, include_corners=False)
    #neighbours_voxels = neighbours_voxels.reshape(-1,3)
    #voxgrid2 = voxgrid2.set_voxel(neighbours_voxels, 80)
    #voxgrid2 = voxgrid2.set_voxel_neighbours(jnp.array([[10, 10, 10], [20, 20, 20]]), 1, 1, include_corners=False)
    #voxgrid2 = voxgrid2.dilate(1)

    #voxgrid = voxgrid.update(voxgrid2)

    #voxgrid = voxgrid.empty()
    #voxgrid = voxgrid.raycast(waypoint_pairs, attrs=attrs)

    #test_waypoints = test_waypoints.reshape(10*4,3)
    #test_voxels = voxgrid.point_to_voxel(test_waypoints)

    #test_voxels = test_voxels[0]
    #voxgrid = voxgrid.set_voxel(test_voxels)
    #test_values = voxgrid.is_voxel_set(test_voxels)
    #print(test_values)

    voxgrid.display_as_o3d(attrmanager)
