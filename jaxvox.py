import dataclasses
import functools
from dataclasses import dataclass

from typing import Union, Tuple

import jax.experimental.sparse
import jax.numpy as jnp

def indexarr2tup(arr: jnp.ndarray) -> Tuple[int, int, int]:
    # NOT JITTABLE!
    return int(arr[0]), int(arr[1]), int(arr[2])

@dataclass
class _VoxelCollection:
    minbound: jnp.ndarray
    maxbound: jnp.ndarray

    padded_grid_dim: int
    real_grid_shape: Tuple[int, int, int]
    padded_grid_shape: Tuple[int, int, int]
    voxel_size: float

    @property
    def padded_error_index_array(self):
        return jnp.array(self.padded_grid_shape) - 1

    @property
    def padded_error_index_tuple(self):
        return (self.padded_grid_shape[0] - 1, self.padded_grid_shape[1] - 1,
                self.padded_grid_shape[2] - 1)  # jnp.array(self.padded_grid_shape) - 1

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def to_voxcol(self):
        return _VoxelCollection(
            minbound=self.minbound,
            maxbound=self.maxbound,
            padded_grid_dim=self.padded_grid_dim,
            real_grid_shape=self.real_grid_shape,
            padded_grid_shape=self.padded_grid_shape,
            voxel_size=self.voxel_size
        )

    @classmethod
    def build_from_bounds(cls, minbound, maxbound, voxel_size=0.05):
        # Compute the grid dimensionsreal_grid_shape
        real_grid_shape = jnp.ceil((maxbound - minbound) / voxel_size).astype(int)

        undo_pad_slicer = indexarr2tup(real_grid_shape)
        min_grid_dim = int(jnp.argmax(real_grid_shape))

        padded_grid_shape = real_grid_shape.at[min_grid_dim].set(real_grid_shape[min_grid_dim] + 1)
        padded_grid_shape = indexarr2tup(padded_grid_shape)

        return _VoxelCollection(minbound=minbound, maxbound=maxbound, padded_grid_dim=min_grid_dim, real_grid_shape=undo_pad_slicer, padded_grid_shape=padded_grid_shape, voxel_size=voxel_size)

    @classmethod
    def build_from_voxcol(cls, voxcol):
        raise NotImplementedError()

    def to_kwargs(self):
        return vars(self)

    @jax.jit
    def point_to_voxel(self, points):
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
    def voxel_to_point(self, voxels):
        voxels = jnp.atleast_2d(voxels)
        return (self.minbound + (voxels + 0.5) * self.voxel_size).squeeze()

    @jax.jit
    def voxel_to_8points(self, voxels):
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


    def add_voxel(self, voxels, attrs=None):
        raise NotImplementedError()

    def is_voxel_set(self, voxel):
        raise NotImplementedError()

    @jax.jit
    def add_point(self, points, attrs=None):
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.add_voxel(voxels, attrs)


    @jax.jit
    def is_point_set(self, points):
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.is_voxel_set(voxels)

    def _cull(self, voxels, attrs=None, do_attrs=None):
        voxels = jnp.atleast_2d(voxels)

        if attrs is None:
            attrs = jnp.ones(voxels.shape[0])
        attrs = attrs.clip(1, jnp.inf)

        voxels = jnp.atleast_2d(voxels)

        def transform_invalid(voxel):
            is_invalid = jnp.logical_or(jnp.any(voxel < 0), jnp.any(voxel >= self.padded_error_index_array)).astype(jnp.int8)
            return self.padded_error_index_array * is_invalid + voxel * (1 - is_invalid), is_invalid

        voxels, invalid = jax.vmap(transform_invalid)(voxels)

        if do_attrs is not None:
            attrs = jax.vmap(lambda a, v: -1*v + a*(1-v))(attrs, invalid)
            return voxels, attrs
        else:
            return voxels

    def _tree_flatten(self):
        children = (self.minbound, self.maxbound)  # arrays / dynamic values
        aux_data = (self.padded_grid_dim, self.real_grid_shape, self.padded_grid_shape, self.voxel_size)  # hashable static values
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        minbound, maxbound = children
        padded_grid_dim, real_grid_shape, padded_grid_shape, voxel_size = aux_data
        return cls(
            minbound=minbound,
            maxbound=maxbound,
            padded_grid_dim=padded_grid_dim,
            real_grid_shape=real_grid_shape,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

    def to_open3d(self):
        return self.to_voxellist().to_open3d()

    def to_voxelgrid(self):
        raise NotImplementedError()

    def to_voxellist(self):
        raise NotImplementedError()

jax.tree_util.register_pytree_node(_VoxelCollection,
                               _VoxelCollection._tree_flatten,
                               _VoxelCollection._tree_unflatten)

import numpy as np

@dataclass
class VoxelGrid(_VoxelCollection):
    _grid: Union[jnp.ndarray, jax.experimental.sparse.BCOO]

    @property
    def grid(self):
        return self._grid[0:self.real_grid_shape[0], 0:self.real_grid_shape[1], 0:self.real_grid_shape[2]]

    @classmethod
    def build_from_bounds(cls, minbound, maxbound, voxel_size=0.05):
        voxcol = super().build_from_bounds(minbound, maxbound, voxel_size)
        return VoxelGrid.build_from_voxcol(voxcol)

    @classmethod
    def build_from_voxcol(cls, voxcol):
        # we want the array to be 0's everywhere where it's possible to set voxels, and -1 in the padding
        grid = jnp.ones(voxcol.padded_grid_shape, dtype=jnp.int32) * -1
        grid = grid.at[0:voxcol.real_grid_shape[0],0:voxcol.real_grid_shape[1],0:voxcol.real_grid_shape[2]].set(0)
        return VoxelGrid(_grid=grid, **vars(voxcol))


    def to_voxellist(self, size=None):
        xs, ys, zs = jnp.nonzero(self.grid, size=size, fill_value=self.padded_error_index_tuple)

        voxels = jnp.concatenate((xs[:,None], ys[:,None], zs[:,None]), axis=1)
        attrs = self.is_voxel_set(voxels)

        voxcol_information = self.to_voxcol()  # VoxelGrid.build_from_voxcol()#.add_voxel(self.voxels, self.attrs)
        voxlist = VoxelList.build_from_voxcol(voxcol_information)
        return voxlist.replace(voxels=voxels, attrs=attrs)# not calling voxlist.add_voxel(voxels, attrs) because this is faster (things are already culled)

    def to_voxelgrid(self):
        return self

    @jax.jit
    def add_voxel(self, voxels, attrs=None):
        voxels = jnp.atleast_2d(voxels)
        voxels, attrs = self._cull(voxels, attrs, do_attrs=True)
        new_grid = self._grid.at[voxels[:,0], voxels[:,1], voxels[:,2]].set(attrs)
        return self.replace(_grid=new_grid)

    @jax.jit
    def is_voxel_set(self, voxels):
        voxels = jnp.atleast_2d(voxels)
        voxels = self._cull(voxels, do_attrs=None)
        return self._grid[voxels[:,0], voxels[:,1], voxels[:,2]].squeeze()

    def _tree_flatten(self):
        children, aux_data = super()._tree_flatten()
        children = (*children, self._grid)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        minbound, maxbound, _grid = children
        padded_grid_dim, real_grid_shape, padded_grid_shape, voxel_size = aux_data
        return cls(
            _grid=_grid,
            minbound=minbound,
            maxbound=maxbound,
            padded_grid_dim=padded_grid_dim,
            real_grid_shape=real_grid_shape,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

jax.tree_util.register_pytree_node(VoxelGrid,
                                   VoxelGrid._tree_flatten,
                                   VoxelGrid._tree_unflatten)



@dataclass
class VoxelList(_VoxelCollection):
    voxels: jnp.ndarray
    attrs: jnp.ndarray

    @classmethod
    def build_from_bounds(cls, minbound, maxbound, voxel_size=0.05):
        voxcol = super().build_from_bounds(minbound, maxbound, voxel_size)
        return VoxelList.build_from_voxcol(voxcol)

    @classmethod
    def build_from_voxcol(cls, voxcol):
        voxels = jnp.array([], dtype=jnp.int32).reshape((0, 3))
        attrs = jnp.array([], dtype=jnp.int32).reshape((0,))
        return VoxelList(voxels=voxels, attrs=attrs, **vars(voxcol))

    def to_voxellist(self):
        return self

    @jax.jit
    def add_voxel(self, voxels, attrs=None):
        voxels = jnp.atleast_2d(voxels)

        voxels, attrs = self._cull(voxels, attrs, do_attrs=True)

        new_voxels = jnp.vstack([self.voxels, voxels])
        new_attrs = jnp.hstack([self.attrs, attrs])

        return self.replace(voxels=new_voxels, attrs=new_attrs)

#    @functools.partial(jax.jit, static_argnames=)
    @jax.jit
    def to_voxelgrid(self) -> VoxelGrid:
        voxcol_information = self.to_voxcol() #VoxelGrid.build_from_voxcol()#.add_voxel(self.voxels, self.attrs)
        voxgrid = VoxelGrid.build_from_voxcol(voxcol_information)
        return voxgrid.add_voxel(self.voxels, self.attrs)


  #  @jax.jit
    def is_voxel_set(self, voxels) -> jnp.ndarray:
        #return self.to_voxelgrid().is_voxel_set(voxels)
        voxels = jnp.atleast_2d(voxels)
        voxels = self._cull(voxels, do_attrs=None)

        def cond(acc):
            return jnp.logical_and(jnp.logical_not(acc[0]), acc[-1]<self.voxels.shape[0])

        def body(acc):
            found, attrs, inputvoxel, selfvoxels, selfattrs, i = acc
            found = jnp.all(selfvoxels[i] == inputvoxel).astype(jnp.int8)
            attrs = selfattrs[i] * found + attrs * (1-found)
            return found, attrs, inputvoxel, selfvoxels, selfattrs, i+1

        def vmap_it(input_voxel, selfvoxel, selfattr):
            acc = (
                jnp.array(0).astype(jnp.int8),
                0,
                input_voxel,
                selfvoxel,
                selfattr,
                0,  # index i
            )

            return jax.lax.while_loop(
                cond,
                body,
                acc
            )[1]
        return jax.vmap(functools.partial(vmap_it, selfvoxel=self.voxels, selfattr=self.attrs))(voxels)


        #return self.to_voxelgrid().is_voxel_set(voxels)

    def _tree_flatten(self):
        children, aux_data = super()._tree_flatten()
        children = (*children, self.voxels, self.attrs)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        minbound, maxbound, voxels, attrs = children
        padded_grid_dim, real_grid_shape, padded_grid_shape, voxel_size = aux_data
        return cls(
            voxels=voxels,
            attrs=attrs,
            minbound=minbound,
            maxbound=maxbound,
            padded_grid_dim=padded_grid_dim,
            real_grid_shape=real_grid_shape,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

    def cull_errors(self):
        # removes all attrs with -1 or whose index is the error index
        # NOT JITTABLE
        return self # todo

    def to_open3d(self):
        self = self.cull_errors()
        centerpoints = self.voxel_to_point(self.voxels)
        attrs = self.attrs

        import open3d as o3d
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(centerpoints)
        #pcd_new.colors = o3d.utility.Vector3dVector(new_colors)
        new_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_new, voxel_size=self.voxel_size)
        return new_grid

jax.tree_util.register_pytree_node(VoxelList,
                                   VoxelList._tree_flatten,
                                   VoxelList._tree_unflatten)

if __name__ == '__main__':
    voxel_grid = VoxelList.build_from_bounds(
        minbound=jnp.array([-0.5, 1.0, -2.5]),
        maxbound=jnp.array([1.0, 10.0, 0.0]),
        voxel_size=0.05,
    )

    point = jnp.array([0.0, 5.0, -1.0])
    voxel_indices_1 = voxel_grid.point_to_voxel(point)
    print(voxel_indices_1)  # Output: [10, 80, 30] (example)

    point = voxel_grid.minbound - 1
    voxel_indices_2 = voxel_grid.point_to_voxel(point)
    print(voxel_indices_2)  # Output: [10, 80, 30] (example)


    voxels = jnp.vstack([voxel_indices_1, voxel_indices_2])
    voxel_grid = voxel_grid.add_voxel(voxels, jnp.array([2,3]))

    print(voxel_grid.is_voxel_set(voxels))  # should be [2, -1]

    print(voxel_grid.is_point_set(voxel_grid.minbound-1))   # should be -1

    voxel_grid = voxel_grid.to_voxelgrid()
    #exit()

    voxel_grid = voxel_grid.to_voxellist(10)


    print(voxel_grid.is_voxel_set(jnp.array([11,81,291])))  # should be -1

    #voxel_grid = voxel_grid.to_voxelgrid()

    voxel_grid = voxel_grid.add_point(jnp.array([0.7, 7.0, -1.0]), jnp.array([28]))
    print(voxel_grid.is_voxel_set(voxel_indices_1)) # should be 2
    print(voxel_grid.is_point_set(jnp.array([0.7, 7.0, -1.0]))) # should be 28


    print(voxel_grid.voxel_to_point(voxel_indices_1))
    print(voxel_grid.voxel_to_8points(voxel_indices_1))

    import open3d as o3d
    o3d_vox = voxel_grid.to_open3d()

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d_vox)
    visualizer.poll_events()
    visualizer.update_renderer()

    view_control = visualizer.get_view_control()
    view_control.set_front([1, 0, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_lookat([0, 0, 0])
    visualizer.run()
    i=0