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
        # useful for subclasses
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

        return _VoxelCollection(
            minbound=minbound,
            maxbound=maxbound,
            padded_grid_dim=min_grid_dim,
            real_grid_shape=undo_pad_slicer,
            padded_grid_shape=padded_grid_shape,
            voxel_size=voxel_size
        )

    @classmethod
    def build_from_voxcol(cls, voxcol):
        # in subclasses, create subclass using info from voxcol
        raise NotImplementedError()

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

    def set_voxel(self, voxels, attrs=None):
        raise NotImplementedError()

    def is_voxel_set(self, voxel):
        raise NotImplementedError()

    def del_voxel(self, voxels):
        raise NotImplementedError()

    def del_point(self, points):
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.del_voxel(voxels)

    @jax.jit
    def set_point(self, points, attrs=None):
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.set_voxel(voxels, attrs)

    @jax.jit
    def is_point_set(self, points):
        points = jnp.atleast_2d(points)
        voxels = self.point_to_voxel(points)  # auto vmap
        return self.is_voxel_set(voxels)

    @jax.jit
    def _cull(self, voxels, attrs=None, do_attrs=None, return_invalid=None):
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

        if do_attrs is not None:
            attrs = jax.vmap(lambda a, v: -1*v + a*(1-v))(attrs, invalid)
            if return_invalid is None:
                return voxels, attrs
            else:
                return voxels, attrs, invalid
        else:
            if return_invalid is None:
                return voxels
            else:
                return voxels, invalid

    def _tree_flatten(self):
        children = (self.minbound, self.maxbound)  # arrays / dynamic values
        aux_data = (self.padded_grid_dim, self.real_grid_shape, self.padded_grid_shape, self.voxel_size)  # hashable static values
        return children, aux_data

    def display_as_o3d(self, attrmanager=None):
        import open3d as o3d
        o3d_vox = self.to_open3d(attrmanager)

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

    def to_voxelgrid(self):
        raise NotImplementedError()

    def to_voxellist(self):
        raise NotImplementedError()

    def to_open3d(self, attrmanager=None):
        return self.to_voxellist().to_open3d(attrmanager)

    @classmethod
    def from_open3d(cls, o3d_grid, import_attrs=False, return_attrmanager=False):
        raise NotImplementedError()

jax.tree_util.register_pytree_node(_VoxelCollection,
                               _VoxelCollection._tree_flatten,
                               _VoxelCollection._tree_unflatten)


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
        grid = jnp.ones(voxcol.padded_grid_shape) * -1
        grid = grid.at[0:voxcol.real_grid_shape[0],0:voxcol.real_grid_shape[1],0:voxcol.real_grid_shape[2]].set(0)
        return VoxelGrid(_grid=grid, **vars(voxcol))

    @jax.jit
    def set_grid(self, grid):
        # grid of unpadded size
        grid = grid.clip(0, jnp.inf)
        new__grid = self._grid.at[0:self.real_grid_shape[0], 0:self.real_grid_shape[1], 0:self.real_grid_shape[2]].set(grid)
        return self.replace(_grid=new__grid)

    def to_voxellist(self, size=None):
        xs, ys, zs = jnp.nonzero(self.grid, size=size, fill_value=self.padded_error_index_tuple)

        voxels = jnp.concatenate((xs[:,None], ys[:,None], zs[:,None]), axis=1)
        attrs = self.is_voxel_set(voxels)

        voxcol_information = self.to_voxcol()  # VoxelGrid.build_from_voxcol()#.add_voxel(self.voxels, self.attrs)
        voxlist = VoxelList.build_from_voxcol(voxcol_information)
        return voxlist.replace(voxels=voxels, attrs=attrs)# not calling voxlist.add_voxel(voxels, attrs) because this is faster (things are already culled)

    def to_voxelgrid(self):
        return self

    #@jax.jit
    def set_voxel(self, voxels, attrs=None):
        voxels = jnp.atleast_2d(voxels)
        voxels, attrs = self._cull(voxels, attrs, do_attrs=True)
        new_grid = self._grid.at[voxels[:,0], voxels[:,1], voxels[:,2]].set(attrs)
        return self.replace(_grid=new_grid)

    def del_voxel(self, voxels):
        voxels = jnp.atleast_2d(voxels)
        return self.set_voxel(voxels, attrs=jnp.zeros(voxels.shape[0]))

    @jax.jit
    def is_voxel_set(self, voxels):
        voxels = jnp.atleast_2d(voxels)
        voxels = self._cull(voxels, do_attrs=None)
        # cull shoots bad voxels to the invalid slice, so handles invalidity implicitly
        return self._grid[voxels[:,0], voxels[:,1], voxels[:,2]].squeeze()

    @classmethod
    def from_open3d(cls, o3d_grid, import_attrs=True, return_attrmanager=False):
        voxlist_and_maybe_attrdict = VoxelList.from_open3d(o3d_grid, import_attrs=import_attrs, return_attrmanager=return_attrmanager)
        if return_attrmanager is None:
            return voxlist_and_maybe_attrdict.to_voxelgrid()
        else:
            return voxlist_and_maybe_attrdict[0].to_voxelgrid(), voxlist_and_maybe_attrdict[1]


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
        attrs = jnp.array([], dtype=jnp.float32).reshape((0,))
        return VoxelList(voxels=voxels, attrs=attrs, **vars(voxcol))

    def to_voxellist(self):
        return self

    #@jax.jit
    def set_voxel(self, voxels, attrs=None):
        # note: this does NOT delete old voxels
        # so the REAL information is the LAST voxel found in the list
        # this is because delete indices is a bit messy in jax, but appending is easy/cheap
        voxels = jnp.atleast_2d(voxels)

        #self = self.del_voxel(voxels)
        voxels, attrs = self._cull(voxels, attrs, do_attrs=True)

        new_voxels = jnp.vstack([self.voxels, voxels])
        new_attrs = jnp.hstack([self.attrs, attrs])

        return self.replace(voxels=new_voxels, attrs=new_attrs)

    def deduplicate(self):
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

#    @functools.partial(jax.jit, static_argnames=)
    @jax.jit
    def to_voxelgrid(self) -> VoxelGrid:
        voxcol_information = self.to_voxcol() #VoxelGrid.build_from_voxcol()#.add_voxel(self.voxels, self.attrs)
        voxgrid = VoxelGrid.build_from_voxcol(voxcol_information)

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

        return voxgrid.set_voxel(rescued_voxels, rescued_attrs)

    #@property
    def _error_pad(self):
        return self.set_voxel(self.padded_error_index_array, -1)

    def del_voxel(self, voxels):
        voxels = jnp.atleast_2d(voxels)
        idxs, attrs = self.find(voxels)

        # janky but it works...
        self = self._error_pad()
        selfvoxels = self.voxels
        selfattrs = self.attrs

        selfvoxels = selfvoxels.at[idxs].set(self.padded_error_index_array)
        selfattrs = selfattrs.at[idxs].set(0)

        return self.replace(
            voxels=selfvoxels[:-1], attrs=selfattrs[:-1]
        )

    def find(self, voxels):
        # returns index and attr of voxel
        # if voxel is not found, both index and attr will be -1
        voxels = jnp.atleast_2d(voxels)
        voxels, invalid = self._cull(voxels, do_attrs=None, return_invalid=True)

        voxshape_minus_ones = jnp.ones(voxels.shape[0], dtype=jnp.uint32) * -1
        carry = (voxels, voxshape_minus_ones, voxshape_minus_ones.astype(jnp.float32))

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
    def is_voxel_set(self, voxels) -> jnp.ndarray:
        return self.find(voxels)[-1]


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

    @jax.jit
    def get_mask_valid(self, voxels=None, attrs=None):
        if voxels is None:
            assert attrs is None
            voxels = self.voxels
            attrs = self.attrs
        return jnp.logical_not(self._cull(voxels, attrs=attrs, return_invalid=True)[-1])

    def to_open3d(self, attrmanager=None):
        mask = self.get_mask_valid()

        voxels = self.voxels[mask]
        attrs = self.attrs[mask]
        centerpoints = self.voxel_to_point(voxels)

        import open3d as o3d
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(centerpoints)

        from jaxvox_attrs import AttrManager
        if attrmanager is not None:
            if isinstance(attrmanager, AttrManager):
                assert isinstance(attrmanager, AttrManager)
                new_colors = attrmanager.get_attrvals_for_attrkeys(attrs, (0,0,0))
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
    def from_open3d(cls, o3d_grid, import_attrs=True, return_attrmanager=False):
        maxbound = o3d_grid.get_max_bound()
        minbound = o3d_grid.get_min_bound()
        voxlist = VoxelList.build_from_bounds(minbound=minbound, maxbound=maxbound, voxel_size=o3d_grid.voxel_size)

        voxels = o3d_grid.get_voxels()  # returns list of voxels
        import numpy as np
        indices = np.stack(list(vx.grid_index for vx in voxels))

        #attrs = []
        if import_attrs is not None and import_attrs is not False:
            from jaxvox_attrs import AttrManager
            color_dict = AttrManager()
            attrs = color_dict.add_attrvals_get_attrkeys([vx.color for vx in voxels])
        else:
            attrs = None

        voxlist = voxlist.set_voxel(voxels=indices, attrs=attrs)

        if import_attrs is not None and import_attrs and return_attrmanager is not None and return_attrmanager:
            return voxlist, color_dict

        return voxlist


jax.tree_util.register_pytree_node(VoxelList,
                                   VoxelList._tree_flatten,
                                   VoxelList._tree_unflatten)

if __name__ == '__main__':
    import open3d as o3d
    import json
    pcd = o3d.io.read_point_cloud("pcd.pcd")

    with open("other_info.json", "r") as f:
        other_info = json.loads(f.read())

    o3d_voxelgrid_from_point_cloud = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.02)

    def test_o3d_io():
        def compare_grids(o3d_grid1, o3d_grid2):
            voxlist1 = VoxelList.from_open3d(o3d_grid1).to_voxelgrid()
            voxlist2 = VoxelList.from_open3d(o3d_grid2).to_voxelgrid()
            return jnp.all(voxlist1.grid == voxlist2.grid)

        voxlist = VoxelList.from_open3d(o3d_voxelgrid_from_point_cloud)

        new_voxel_grid = voxlist.to_open3d()

        voxlist = VoxelList.from_open3d(new_voxel_grid)

        voxgrid = voxlist.to_voxelgrid()

        og_voxel_grid = voxgrid.to_open3d()

        assert compare_grids(og_voxel_grid, new_voxel_grid)

    def test_display():
        voxgrid, attr_mapping = VoxelGrid.from_open3d(o3d_voxelgrid_from_point_cloud, return_attrmanager=True)

        inner_grid = voxgrid.grid
        inner_grid = inner_grid.at[:,0,:].set(1)
        voxgrid = voxgrid.set_grid(inner_grid)

        #voxgrid = voxgrid.add_voxel()

        import matplotlib.pyplot as plt
        attr_mapping = attr_mapping #plt.get_cmap("gist_rainbow") # values to try: attr_mapping, None, and a colormap

        voxgrid.display_as_o3d(attr_mapping)

    def test_some_io():
        voxlist = VoxelList.build_from_bounds(
            minbound=jnp.array([-0.5, 1.0, -2.5]),
            maxbound=jnp.array([1.0, 10.0, 0.0]),
            voxel_size=0.05,
        )

        point = jnp.array([0.0, 5.0, -1.0])
        voxel_indices_1 = voxlist.point_to_voxel(point)
        #print(voxel_indices_1)  # Output: [10, 80, 30] (example)

        point = voxlist.minbound - 1
        voxel_indices_2 = voxlist.point_to_voxel(point)
        #print(voxel_indices_2)  # Output: [10, 80, 30] (example)

        voxels = jnp.vstack([voxel_indices_1, voxel_indices_2])
        voxlist = voxlist.set_voxel(voxels, jnp.array([2, 3]))

        assert jnp.all(voxlist.is_voxel_set(voxels) == jnp.array([2, -1]))
        assert jnp.all(voxlist.is_point_set(voxlist.minbound - 1) == -1)

        voxlist = voxlist.set_voxel(voxels, jnp.array([23, 3]))
        assert jnp.all(voxlist.is_voxel_set(voxels) == jnp.array([23, -1]))

        #voxlist = voxlist.deduplicate()


        voxel_grid = voxlist.to_voxelgrid()
        # exit()

        #voxel_grid = voxel_grid.to_voxellist(10)

        assert jnp.all(voxel_grid.is_voxel_set(jnp.array([11, 81, 291])) == -1)

        #print())  # should be -1

        # voxel_grid = voxel_grid.to_voxelgrid()

        voxel_grid = voxel_grid.set_point(jnp.array([0.7, 7.0, -1.0]), jnp.array([28]))

        # todo how do i ensure
        assert jnp.all(voxel_grid.is_voxel_set(voxel_indices_1) == 23)
        assert jnp.all(voxel_grid.is_point_set(jnp.array([0.7, 7.0, -1.0])) == 28)

        voxel_grid = voxel_grid.del_point(jnp.array([0.7, 7.0, -1.0]))

        #print(voxel_grid.voxel_to_point(voxel_indices_1))
        #print(voxel_grid.voxel_to_8points(voxel_indices_1))

    #test_o3d_io()
    #test_display()
    test_some_io()


    exit()
    """



    

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
    """