
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Tuple, Union

import math
from typing_extensions import Self

import jax
import jax.experimental
import jax.lax
from jax import numpy as jnp

from jaxvox.jaxutils import bool_ifelse
from jaxvox.jaxvox_attrs import AttrManager


def indexarr2tup(arr: jnp.ndarray, type) -> Tuple:
    # NOT JITTABLE!
    return type(arr[0]), type(arr[1]), type(arr[2])





@dataclass
class VoxCol:
    _minbound: Tuple[float, float, float] #jnp.ndarray
    _maxbound: Tuple[float, float, float] #jnp.ndarray

    padded_grid_dim: int
    real_grid_shape: Tuple[int, int, int]
    padded_grid_shape: Tuple[int, int, int]
    voxel_size: float

    @property
    def minbound(self) -> jnp.ndarray:
        return jnp.array(self._minbound)

    @property
    def maxbound(self) -> jnp.ndarray:
        return jnp.array(self._maxbound)

    @property
    def padded_error_index_array(self) -> jnp.array:
        return jnp.array(self.padded_grid_shape) - 1

    @property
    def padded_error_index_tuple(self) -> Tuple[int, int, int]:
        return (self.padded_grid_shape[0] - 1, self.padded_grid_shape[1] - 1,
                self.padded_grid_shape[2] - 1)  # jnp.array(self.padded_grid_shape) - 1

    def replace(self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)

    def to_voxcol(self) -> VoxCol:
        # useful for subclasses
        return VoxCol(
            _minbound=self._minbound,
            _maxbound=self._maxbound,
            padded_grid_dim=self.padded_grid_dim,
            real_grid_shape=self.real_grid_shape,
            padded_grid_shape=self.padded_grid_shape,
            voxel_size=self.voxel_size
        )

    @classmethod
    def build_from_bounds(cls, minbound: jnp.array, maxbound: jnp.array, voxel_size: float=0.05) -> VoxCol:
        # Compute the grid dimensionsreal_grid_shape
        real_grid_shape = jnp.ceil((maxbound - minbound) / voxel_size).astype(int)

        undo_pad_slicer = indexarr2tup(real_grid_shape, int)
        min_grid_dim = int(jnp.argmax(real_grid_shape))

        padded_grid_shape = real_grid_shape.at[min_grid_dim].set(real_grid_shape[min_grid_dim] + 1)
        padded_grid_shape = indexarr2tup(padded_grid_shape, int)

        return VoxCol(
            _minbound=indexarr2tup(minbound, float),
            _maxbound=indexarr2tup(maxbound, float),
            padded_grid_dim=min_grid_dim,
            real_grid_shape=undo_pad_slicer,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

    @classmethod
    def build_from_voxcol(cls, voxcol) -> Self:
        # in subclasses, create subclass using info from voxcol
        raise NotImplementedError()

    #@jax.jit
    def point_to_voxel(self, points) -> jnp.ndarray:
        @jax.jit
        def _point_to_voxel(self, point):
            point_is_within_bounds = jnp.all(self.minbound <= point).astype(jnp.int8) + jnp.all(
                point <= self.maxbound).astype(jnp.int8)
            point_is_within_bounds = (point_is_within_bounds == 2).astype(jnp.int8)

            normalized_point = (point - self.minbound) / self.voxel_size

            voxel_indices = jnp.floor(normalized_point).astype(int)

            voxel_indices = point_is_within_bounds * voxel_indices + (1 - point_is_within_bounds) * (
                self.padded_error_index_array)

            return voxel_indices
        points = jnp.atleast_2d(points)

        return jax.vmap(functools.partial(_point_to_voxel, self))(points).squeeze()

    @jax.jit
    def voxel_to_point(self, voxels) -> jnp.ndarray:
        voxels = jnp.atleast_2d(voxels)
        return (self.minbound + (voxels + 0.5) * self.voxel_size).squeeze()

    @jax.jit
    def voxel_to_8points(self, voxels) -> jnp.ndarray:
        voxels = jnp.atleast_2d(voxels)
        @jax.jit
        def _voxel_to_8points(self, voxel):
            # Compute the coordinates of the voxel's origin (minimum corner)
            voxel_origin = self.minbound + voxel * self.voxel_size

            # Compute the offsets for the 8 corners
            offsets = jnp.array([
                [0, 0, 0],  # Bottom-front-left
                [1, 0, 0],  # Bottom-front-right
                [0, 1, 0],  # Bottom-back-left
                [1, 1, 0],  # Bottom-back-right
                [0, 0, 1],  # Top-front-left
                [1, 0, 1],  # Top-front-right
                [0, 1, 1],  # Top-back-left
                [1, 1, 1],  # Top-back-right
            ]) * self.voxel_size

            # Add offsets to the voxel's origin to compute the corner points
            corner_points = voxel_origin + offsets

            return corner_points
        return jax.vmap(functools.partial(_voxel_to_8points, self))(voxels).squeeze()

    def set_voxel(self, voxels, attrs=None) -> Self:
        raise NotImplementedError()

    def is_voxel_set(self, voxel) -> jnp.ndarray:
        raise NotImplementedError()

    @functools.partial(jax.jit, static_argnums=(2,3))
    def voxel_to_neighbours(self, voxel, distance:int=1, include_corners:bool=False):
        voxel = jnp.atleast_2d(voxel)

        offset_range = jnp.arange(-distance, distance + 1)
        dx, dy, dz = jnp.meshgrid(offset_range, offset_range, offset_range, indexing="ij")
        offsets = jnp.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)

        if include_corners:
            # chebyshev
            # this is just the full mask, i believe
            NUM_NEIGHBOURS = offsets.shape[0]
            MASK = jnp.ones(offsets.shape[0], dtype=jnp.uint32)
        else:
            # manhattan
            NUM_NEIGHBOURS = int(6*distance*(distance+1)/2+1)
            MASK = jnp.sum(jnp.abs(offsets), axis=1) == distance
            MASK = MASK + (jnp.sum(jnp.abs(offsets), axis=1) == 0)

        valid_offsets = offsets * MASK.astype(jnp.uint32)[:, None]
        valid_offset_indices = jnp.nonzero(MASK, size=NUM_NEIGHBOURS,
                                           fill_value=2 * offsets.shape[0])
        valid_offsets = valid_offsets[valid_offset_indices]

        def do_each_voxel(valid_offsets, voxel):
            return voxel + valid_offsets

        neighbours = jax.vmap(functools.partial(do_each_voxel, valid_offsets))(voxel)
        return neighbours

    @functools.partial(jax.jit, static_argnums=(2, 3))
    def point_to_neighbours(self, points, distance:int=1, include_corners:bool=False):
        voxel = self.point_to_voxel(points)
        return self.voxel_to_neighbours(voxel, distance, include_corners)

    @functools.partial(jax.jit, static_argnums=(3, 4))
    def set_voxel_neighbours(self, voxels, attrs:int=None, distance:int=1, include_corners:bool=False):
        neighbours = self.voxel_to_neighbours(voxels, distance, include_corners)
        neighbours = neighbours.reshape(-1, 3)
        return self.set_voxel(neighbours, attrs)

    @functools.partial(jax.jit, static_argnums=(3, 4))
    def set_point_neighbours(self, points, attrs:int=None, distance:int=1, include_corners:bool=False):
        voxel = self.point_to_voxel(points)
        return self.set_voxel_neighbours(voxel, attrs, distance, include_corners)

    @jax.jit
    def empty(self):
        return self

    def del_voxel(self, voxels) -> Self:
        voxels = jnp.atleast_2d(voxels)
        return self.set_voxel(voxels, attrs=jnp.zeros(voxels.shape[0]))

    def del_point(self, points) -> Self:
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.del_voxel(voxels)

    @jax.jit
    def set_point(self, points, attrs=None) -> Self:
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.set_voxel(voxels, attrs)

    @jax.jit
    def is_point_set(self, points) -> jnp.ndarray:
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.is_voxel_set(voxels)

    @functools.partial(jax.jit, static_argnums=(3,4))
    def _cull(self, voxels, attrs=None, return_attrs:bool=False, return_invalid:bool=False) -> Union[jnp.array, Tuple[jnp.array, jnp.array], Tuple[jnp.array, jnp.array, jnp.array]]:
        voxels = jnp.atleast_2d(voxels)

        if attrs is None:
            attrs = jnp.ones(voxels.shape[0])
        else:
            attrs = jnp.atleast_1d(attrs)

        def transform_invalid(voxel):
            out_low_bound = jnp.any(voxel < 0)
            out_high_bound = jnp.logical_or(voxel[self.padded_grid_dim] >= self.padded_error_index_array[self.padded_grid_dim], jnp.any(voxel > self.padded_error_index_array))

            is_invalid = jnp.logical_or(out_low_bound, out_high_bound).astype(jnp.int8)
            return self.padded_error_index_array * is_invalid + voxel * (1 - is_invalid), is_invalid

        voxels, invalid = jax.vmap(transform_invalid)(voxels)

        if return_attrs is False:
            if not return_invalid:
                return voxels
            else:
                return voxels, invalid
        else:
            attrs = jax.vmap(lambda a, v: -1*v + a*(1-v))(attrs, invalid)
            if not return_invalid:
                return voxels, attrs
            else:
                return voxels, attrs, invalid


    def _tree_flatten(self):
        children = tuple()  # arrays / dynamic values
        aux_data = (self._minbound, self._maxbound, self.padded_grid_dim, self.real_grid_shape, self.padded_grid_shape, self.voxel_size)  # hashable static values
        return children, aux_data

    def display_as_o3d(self, attrmanager=None) -> None:
        import open3d as o3d
        o3d_vox = self.to_open3d(attrmanager)

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(o3d_vox)
        visualizer.poll_events()
        visualizer.update_renderer()

        visualizer.run()

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        minbound, maxbound, padded_grid_dim, real_grid_shape, padded_grid_shape, voxel_size = aux_data
        return cls(
            _minbound=minbound,
            _maxbound=maxbound,
            padded_grid_dim=padded_grid_dim,
            real_grid_shape=real_grid_shape,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

    def get_masked_voxel(self, voxels, mask):
        # returns the ``error voxel`` for false masks, the correct voxel otherwise
        return bool_ifelse(mask, voxels, self.padded_error_index_array)

    def to_voxelgrid(self) -> VoxGrid:
        raise NotImplementedError()

    def to_voxellist(self) -> VoxList:
        raise NotImplementedError()

    def to_open3d(self, attrmanager=None):
        return self.to_voxellist().to_open3d(attrmanager)

    @classmethod
    def from_open3d(cls, o3d_grid, import_attrs=True, return_attrmanager=False) -> Self:
        maxbound = o3d_grid.get_max_bound()
        minbound = o3d_grid.get_min_bound()
        return VoxCol.build_from_bounds(minbound=minbound, maxbound=maxbound, voxel_size=o3d_grid.voxel_size)

    @property
    def _raycasting_worst_case_num_steps(self):
        # disgusting... but we cant use arrays if we want to be static -_-
        diff = (
            self._maxbound[0] - self._minbound[0],
            self._maxbound[1] - self._minbound[1],
            self._maxbound[2] - self._minbound[2]
        )
        diff = (diff[0] ** 2, diff[1] ** 2, diff[2] ** 2)
        summed = diff[0] + diff[1] + diff[2]
        squared = math.sqrt(summed)
        return int(squared / self.voxel_size)

    def _calc_ray_points_for_pair(self, x, y):
        direction = y - x
        direction = direction / jnp.linalg.norm(direction)

        distance = jnp.linalg.norm(y - x)
        distance_per_step = distance / self._raycasting_worst_case_num_steps
        steps = jnp.arange(self._raycasting_worst_case_num_steps) * distance_per_step

        def mul(direction, step):
            return direction * step

        step_points = jax.vmap(functools.partial(mul, direction))(steps)  # * direction
        step_points = step_points + x
        return jnp.concatenate([step_points, y[None]], axis=0)

    @functools.partial(jax.jit, static_argnums=(4,))
    def raycast(self, x_points, y_points=None, attrs=None, return_aux: bool=False):
        if attrs is None:
            attrs = 1
        attrs = jnp.atleast_1d(attrs)

        if len(x_points.shape) == 4 and y_points is None:
            y_points = x_points[:,1]
            x_points = x_points[:,0]

            ORIG_POINT_SHAPE = (x_points.shape[0], x_points.shape[1])

            if attrs is not None and len(attrs.shape) == 2:
                pass
            else:
                if attrs.shape[0] == x_points.shape[0]:
                    attrs = jnp.repeat(attrs[:,None], ORIG_POINT_SHAPE[1], axis=1)
                else:
                    attrs = jnp.repeat(attrs[None], ORIG_POINT_SHAPE[0], axis=0)

            x_points = x_points.reshape(x_points.shape[0] * x_points.shape[1], 3)
            y_points = y_points.reshape(y_points.shape[0] * y_points.shape[1], 3)

        elif len(x_points.shape) == 3 and y_points is None:
            y_points = x_points[1]
            x_points = x_points[0]

            ORIG_POINT_SHAPE = (x_points.shape[0],)
        else:
            ORIG_POINT_SHAPE = (x_points.shape[0],)

        x_centers = self.voxel_to_point(self.point_to_voxel(x_points))
        y_centers = self.voxel_to_point(self.point_to_voxel(y_points))

        rays = jax.vmap(self._calc_ray_points_for_pair)(x_centers, y_centers)

        final_attr_shape = jnp.ones((*ORIG_POINT_SHAPE, self._raycasting_worst_case_num_steps+1))

        if attrs.shape[0] == 1 and len(attrs.shape) == 1:
            final_attrs = attrs * final_attr_shape
        else:
            if len(ORIG_POINT_SHAPE) == 2:
                def attr_mul(a, mat):
                    return a[:,None] * mat
            else:
                def attr_mul(a, mat):
                    return a * mat
            final_attrs = jax.vmap(attr_mul)(attrs, final_attr_shape)

        ray_voxels = self.point_to_voxel(rays.reshape(rays.shape[0] * rays.shape[1], 3))
        ray_voxels_attrs = final_attrs.flatten()

        self = self.set_voxel(ray_voxels, ray_voxels_attrs)

        if not return_aux:
            return self
        else:
            aux = (
                rays.reshape(*ORIG_POINT_SHAPE, self._raycasting_worst_case_num_steps + 1, 3),
                ray_voxels.reshape(*ORIG_POINT_SHAPE, self._raycasting_worst_case_num_steps + 1, 3),
                final_attrs,
            )


            # (ray_points, ray_voxels, ray_voxels_attrs)
            return self, aux



    @jax.jit
    def __raycast(self, x_points, y_points=None, attrs=None):
        if y_points is None:
            y_points = x_points[1]
            x_points = x_points[0]

        if len(x_points.shape) == 2:
            x_points = x_points[None]
            y_points = y_points[None]

        def _raycast_one(self, x_points, y_points=None, attrs=None):

            if attrs is None:
                attrs = jnp.ones(x_points.shape[0])

            x_centers = self.voxel_to_point(self.point_to_voxel(x_points))
            y_centers = self.voxel_to_point(self.point_to_voxel(y_points))

            def _calc_ray(self, x, y):
                direction = y - x
                direction = direction / jnp.linalg.norm(direction)

                distance = jnp.linalg.norm(y - x)
                distance_per_step = distance / self._raycasting_worst_case_num_steps
                steps = jnp.arange(self._raycasting_worst_case_num_steps + 1) * distance_per_step

                def mul(direction, step):
                    return direction * step

                step_points = jax.vmap(functools.partial(mul, direction))(steps)  # * direction
                return step_points + x

            rays = jax.vmap(functools.partial(_calc_ray, self))(x_centers, y_centers)

            ray_attrs = jnp.ones((x_points.shape[0], rays.shape[1]))
            ray_attrs = ray_attrs * attrs[:, None]

            rays = rays.reshape(x_points.shape[0] * rays.shape[1], 3)
            ray_attrs = ray_attrs.reshape(x_points.shape[0] * ray_attrs.shape[1])
            return rays, ray_attrs
            return self.set_point(rays, ray_attrs)

        #x_points = jnp.atleast_3d(x_points)
        #y_points = jnp.atleast_3d(y_points)

        rays, rays_attrs = jax.vmap(functools.partial(_raycast_one, self))( x_points, y_points, attrs)

        rays = rays.squeeze()
        rays_attrs = rays_attrs.squeeze()
        return self.set_point(rays, rays_attrs)




jax.tree_util.register_pytree_node(VoxCol,
                                   VoxCol._tree_flatten,
                                   VoxCol._tree_unflatten)

import jaxvox.voxgrid_at_utils as voxgrid_at_utils
import jaxvox.voxlist_at_utils as voxlist_at_utils

@dataclass
class VoxGrid(VoxCol):
    grid: jnp.ndarray

    @classmethod
    def build_from_bounds(cls, minbound: jnp.array, maxbound:jnp.array, voxel_size: float=0.05) -> Self:
        voxcol = super().build_from_bounds(minbound, maxbound, voxel_size)
        return VoxGrid.build_from_voxcol(voxcol)

    @classmethod
    def build_from_voxcol(cls, voxcol: VoxCol) -> Self:
        # we want the array to be 0's everywhere where it's possible to set voxels, and -1 in the padding
        grid = jnp.zeros(voxcol.real_grid_shape)
        #grid = grid.at[0:voxcol.real_grid_shape[0],0:voxcol.real_grid_shape[1],0:voxcol.real_grid_shape[2]].set(0)
        return VoxGrid(grid=grid, **vars(voxcol))

    @jax.jit
    def empty(self):
        return VoxGrid.build_from_voxcol(self.to_voxcol())

    def attrs_to_1(self):
        return self.replace(grid=(self.grid >= 1).astype(jnp.float32))

    def has_collision(self, voxgrid2):
        return (self.attrs_to_1().grid + voxgrid2.attrs_to_1().grid >= 2).any()

    @jax.jit
    def update(self, grid):
        if isinstance(grid, VoxList):
            grid = grid.to_voxelgrid()
        if isinstance(grid, VoxGrid):
            grid = grid.grid

        mask = (grid > 0).astype(jnp.uint32)
        new_grid = self.grid * (1-mask) + grid * mask
        return self.replace(grid=new_grid)

    """
    #TODO can we get away from the error slice by using proper .at functionality?
    @jax.jit
    def set_grid(self, grid: jnp.array) -> Self:
        # grid of unpadded size
        grid = grid.clip(0, jnp.inf)
        new__grid = self.grid.at[0:self.real_grid_shape[0], 0:self.real_grid_shape[1], 0:self.real_grid_shape[2]].set(
            grid[0:self.real_grid_shape[0], 0:self.real_grid_shape[1], 0:self.real_grid_shape[2]],
            mode="drop"
        )
        return self.replace(_grid=new__grid)"""

    def to_voxellist(self, size: int=None) -> VoxList:
        xs, ys, zs = jnp.nonzero(self.grid, size=size, fill_value=self.padded_error_index_tuple)

        voxels = jnp.concatenate((xs[:,None], ys[:,None], zs[:,None]), axis=1)
        attrs = self.is_voxel_set(voxels)

        voxcol_information = self.to_voxcol()  # VoxelGrid.build_from_voxcol()#.add_voxel(self.voxels, self.attrs)
        voxlist = VoxList.build_from_voxcol(voxcol_information)
        return voxlist.set_voxel(voxels, attrs)# not calling voxlist.add_voxel(voxels, attrs) because this is faster (things are already culled)

    def to_voxelgrid(self) -> Self:
        return self

    #@jax.jit
    def set_voxel(self, voxels: jnp.array, attrs=None) -> Self:
        voxels = jnp.atleast_2d(voxels)

        if attrs is None:
            attrs = jnp.ones(voxels.shape[0])

        return self.at[voxels[:, 0], voxels[:, 1], voxels[:, 2]].set(attrs)

    @functools.partial(jax.jit, static_argnums=(1,))
    def dilate(self, distance=1, attr=None) -> Self:
        # very costly! use carefully
        # in most applications, it might be more worth it to call .dilate(1).dilate(1) vs .dilate(2) !

        if attr is None:
            attr = 1

        kernel_shape = distance + distance + 1

        kernel = jnp.ones((kernel_shape, kernel_shape, kernel_shape), dtype=int)
        new_grid = jax.scipy.signal.convolve(self.grid, kernel, mode='same')

        diff = (new_grid != self.grid).astype(jnp.uint32)
        dilated_grid = self.grid * (1-diff) + attr * diff

        return self.replace(grid=dilated_grid)

    @property
    def at(self):
        return voxgrid_at_utils._VoxGridIndexUpdateHelper(self)

    @jax.jit
    def is_voxel_set(self, voxels: jnp.array) -> jnp.array:
        voxels = jnp.atleast_2d(voxels)
        return self.at[voxels[:,0], voxels[:,1], voxels[:,2]].get()

    def __add__(self, other: VoxGrid):
        if isinstance(other, VoxGrid):
            other = other.grid
        return self.replace(grid=self.grid+ other)


    def __sub__(self, other: VoxGrid):
        if isinstance(other, VoxGrid):
            other = other.grid
        return self.replace(grid=self.grid - other)

    def __mul__(self, other: VoxGrid):
        if isinstance(other, VoxGrid):
            other = other.grid
        return self.replace(grid=self.grid * other)

    def __floordiv__(self, other: VoxGrid):
        if isinstance(other, VoxGrid):
            other = other.grid
        return self.replace(grid=jnp.floor_divide(self.grid, other))

    def __truediv__(self, other: VoxGrid):
        if isinstance(other, VoxGrid):
            other = other.grid
        return self.replace(grid=self.grid / other)

    @classmethod
    def from_open3d(cls, o3d_grid, import_attrs=True, return_attrmanager=False) -> Self:
        voxlist_and_maybe_attrdict = VoxList.from_open3d(o3d_grid, import_attrs=import_attrs, return_attrmanager=return_attrmanager)
        if return_attrmanager is None:
            return voxlist_and_maybe_attrdict.to_voxelgrid()
        else:
            return voxlist_and_maybe_attrdict[0].to_voxelgrid(), voxlist_and_maybe_attrdict[1]


    def _tree_flatten(self):
        children, aux_data = super()._tree_flatten()
        children = (self.grid,)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        _grid = children[0]
        minbound, maxbound, padded_grid_dim, real_grid_shape, padded_grid_shape, voxel_size = aux_data
        return cls(
            grid=_grid,
            _minbound=minbound,
            _maxbound=maxbound,
            padded_grid_dim=padded_grid_dim,
            real_grid_shape=real_grid_shape,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

jax.tree_util.register_pytree_node(VoxGrid,
                                   VoxGrid._tree_flatten,
                                   VoxGrid._tree_unflatten)


@dataclass
class VoxList(VoxCol):
    voxels: jnp.ndarray
    attrs: jnp.ndarray

    @classmethod
    def build_from_bounds(cls, minbound: jnp.array, maxbound:jnp.array, voxel_size:float=0.05) -> Self:
        voxcol = super().build_from_bounds(minbound, maxbound, voxel_size)
        return VoxList.build_from_voxcol(voxcol)

    @jax.jit
    def empty(self):
        return VoxList.build_from_voxcol(self.to_voxcol())

    @classmethod
    def build_from_voxcol(cls, voxcol: VoxCol) -> Self:
        voxels = jnp.array([], dtype=jnp.int32).reshape((0, 3))
        attrs = jnp.array([], dtype=jnp.float32).reshape((0,))
        return VoxList(voxels=voxels, attrs=attrs, **vars(voxcol))

    def to_voxellist(self) -> Self:
        return self

    #@jax.jit
    def set_voxel(self, voxels: jnp.array, attrs=None) -> jnp.array:
        # note: this does NOT delete old voxels
        # so the REAL information is the LAST voxel found in the list
        # this is because delete indices is a bit messy in jax, but appending is easy/cheap
        # I repeat: need to take into account that THE LAST INFO IN selt.voxels and self.attrs is the real, good, info!
        voxels = jnp.atleast_2d(voxels)

        #self = self.del_voxel(voxels)
        voxels, attrs = self._cull(voxels, attrs, return_attrs=True)

        new_voxels = jnp.vstack([self.voxels, voxels])
        new_attrs = jnp.hstack([self.attrs, attrs])

        return self.replace(voxels=new_voxels, attrs=new_attrs)

    def deduplicate(self) -> Self:
        # NOT JITTABLE
        # traverses the list of voxels and removes duplicates. Only keeps last copy in list.
        idxs, _ = self.find(self.voxels)
        idxs = jnp.unique(idxs)
        self = self._error_pad()

        selfvoxels = self.voxels[idxs]
        selfattrs = self.attrs[idxs]

        attr_is_0_or_minus_1 = jnp.logical_or(selfattrs == 0, selfattrs == -1)
        attr_is_valid = jnp.logical_not(attr_is_0_or_minus_1)
        selfvoxels = selfvoxels[attr_is_valid]
        selfattrs = selfattrs[attr_is_valid]
        return self.replace(voxels=selfvoxels, attrs=selfattrs)

    #@jax.jit
    def update(self, grid):
        # must be a voxgrid or voxlist
        if isinstance(grid, VoxGrid):
            grid = grid.to_voxellist()

        return self.set_voxel(grid.voxels, grid.attrs)

#    @functools.partial(jax.jit, static_argnames=)
    @jax.jit
    def to_voxelgrid(self) -> VoxGrid:
        # this ensures that the LAST information is the one that is actually taken into account.
        # So this basically does "self.deduplication" on the fly,
        # except it doesnt remove voxels that are set to 0, because they don't matter since the VoxelGrid has all-0
        # values by default anyway
        # Why not use self.deduplicate? Because this way, we can jit :)
        original_voxels = self.voxels
        original_attrs = self.attrs

        all_idxs = jnp.arange(original_voxels.shape[0], dtype=jnp.uint32)
        killed_voxels = original_voxels.at[all_idxs].set(self.padded_error_index_array)
        killed_attrs = original_attrs.at[all_idxs].set(-1)

        idxs, _ = self.find(original_voxels)

        rescued_voxels = killed_voxels.at[idxs].set(original_voxels[idxs])
        rescued_attrs = killed_attrs.at[idxs].set(original_attrs[idxs])

        voxcol_information = self.to_voxcol()
        voxgrid = VoxGrid.build_from_voxcol(voxcol_information)
        return voxgrid.set_voxel(rescued_voxels, rescued_attrs)

    #@property
    def _error_pad(self) -> Self:
        return self.set_voxel(self.padded_error_index_array, -1)

    def find(self, voxels) -> Tuple[jnp.array, jnp.array]:
        # returns index and attr of voxel
        # if voxel is not found, index will be -1, but attr will be 0 if valid and -1 if invalid
        voxels = jnp.atleast_2d(voxels)
        voxels, invalid = self._cull(voxels, return_attrs=False, return_invalid=True)

        voxshape_minus_ones = jnp.ones(voxels.shape[0], dtype=jnp.uint32) * -1
        carry = (voxels, voxshape_minus_ones, jnp.zeros_like(voxshape_minus_ones,dtype=jnp.float32))

        def scan_func(carry, vox_attr):
            seeking_voxels, seek_vox_idx, seek_vox_attr = carry
            scanning_idx, scanning_vox, scanning_attr = vox_attr

            is_same_vox_mask = jnp.all(seeking_voxels == scanning_vox, axis=1).astype(jnp.uint32)
            seek_vox_idx = seek_vox_idx * (1-is_same_vox_mask) + (jnp.ones_like(seek_vox_idx) * scanning_idx) * is_same_vox_mask
            seek_vox_attr = seek_vox_attr * (1-is_same_vox_mask) + (jnp.ones_like(seek_vox_attr) * scanning_attr) * is_same_vox_mask

            return (seeking_voxels, seek_vox_idx, seek_vox_attr), 1

        (_, seek_vox_idx, seek_vox_attr), _ = jax.lax.scan(scan_func, carry, (jnp.arange(self.voxels.shape[0], dtype=jnp.uint32), self.voxels, self.attrs))

        seek_vox_idx = (-1) * invalid + seek_vox_idx * (1-invalid)  # technically don't need to do this
        seek_vox_attr = (-1) * invalid + seek_vox_attr * (1-invalid)
        return seek_vox_idx, seek_vox_attr

    @jax.jit
    def is_voxel_set(self, voxels: jnp.array) -> jnp.ndarray:
        return self.find(voxels)[-1]

    @property
    def at(self):
        raise NotImplementedError()
        return voxlist_at_utils._VoxListIndexUpdateHelper(self)

    def _tree_flatten(self):
        children, aux_data = super()._tree_flatten()
        children = (self.voxels, self.attrs)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        voxels, attrs = children
        minbound, maxbound, padded_grid_dim, real_grid_shape, padded_grid_shape, voxel_size = aux_data
        return cls(
            voxels=voxels,
            attrs=attrs,
            _minbound=minbound,
            _maxbound=maxbound,
            padded_grid_dim=padded_grid_dim,
            real_grid_shape=real_grid_shape,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

    @jax.jit
    def get_mask_valid(self, voxels: jnp.array=None, attrs: jnp.array=None) -> jnp.array:
        if voxels is None:
            assert attrs is None
            voxels = self.voxels
            attrs = self.attrs
        return jnp.logical_not(self._cull(voxels, attrs=attrs, return_invalid=True)[-1])

    def to_open3d(self, attrmanager=None):
        self = self.deduplicate()
        mask = self.get_mask_valid()

        voxels = self.voxels[mask]
        attrs = self.attrs[mask]
        centerpoints = self.voxel_to_point(voxels)

        import open3d as o3d
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(centerpoints)

        if attrmanager is not None:
            if isinstance(attrmanager, AttrManager):
                assert isinstance(attrmanager, AttrManager)
                new_colors = attrmanager.get_attrvals_for_attrkeys(attrs)
            else:
                import matplotlib
                if isinstance(attrmanager, matplotlib.colors.Colormap):
                    cmap = attrmanager
                    # assuming it is a cmap
                    attrs = attrs - attrs.min()
                    attrs = attrs / attrs.max()
                    new_colors = cmap(attrs)[:,:-1]
                else:
                    raise NotImplementedError()

            pcd_new.colors = o3d.utility.Vector3dVector(new_colors)

        new_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(pcd_new, voxel_size=self.voxel_size, min_bound=self.minbound, max_bound=self.maxbound)
        return new_grid

    @classmethod
    def from_open3d(cls, o3d_grid, import_attrs=True, return_attrmanager=False) -> Self:
        voxcol = super().from_open3d(o3d_grid) #.build_from_bounds(minbound=minbound, maxbound=maxbound, voxel_size=o3d_grid.voxel_size)
        voxlist = VoxList.build_from_voxcol(voxcol)

        voxels = o3d_grid.get_voxels()  # returns list of voxels
        import numpy as np
        indices = np.stack(list(vx.grid_index for vx in voxels))

        #attrs = []
        if import_attrs is not None and import_attrs is not False:
            color_dict = AttrManager()
            attrs = color_dict.add_attrvals_get_attrkeys([vx.color for vx in voxels])
        else:
            attrs = None

        voxlist = voxlist.set_voxel(voxels=indices, attrs=attrs)

        if import_attrs is not None and import_attrs and return_attrmanager is not None and return_attrmanager:
            return voxlist, color_dict

        return voxlist

jax.tree_util.register_pytree_node(VoxList,
                                   VoxList._tree_flatten,
                                   VoxList._tree_unflatten)
