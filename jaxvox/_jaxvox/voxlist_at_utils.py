import jax.numpy as jnp
from jaxvox._jaxvox.slicer import concretize_index


# TODO WIPPPPP



class _VoxListIndexUpdateHelper:

    __slots__ = ("voxlist",)
    def __init__(self, voxlist):
        self.voxlist = voxlist

    def __getitem__(self, index):
        return _IndexUpdateRef(self.voxlist, index)

    def __repr__(self):
        return f"_IndexUpdateHelper({repr(self.voxlist)})"

#def concretize_index(voxcol, index):

class _IndexUpdateRef:
  __slots__ = ("voxlist", "index")

  def __init__(self, voxlist, index):
    self.voxlist = voxlist
    self.index = concretize_index(index)

  def __repr__(self):
    return f"_IndexUpdateRef({repr(self.voxlist)}, {repr(self.index)})"

  def get(self):
      return self.voxlist.is_voxel_set(self.index)
      #return self.voxlist.grid.at[self.index].get(indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, fill_value=-1, mode="fill")

  def _prep_generic_setter(self):
      error_padded_voxlist = self.voxlist._error_pad()
      culled_voxels = error_padded_voxlist._cull(self.index)
      # now, the OOB voxels are the error_index
      _, attrs = error_padded_voxlist.find(culled_voxels)
      # unfortunately, the idxs returned by find are -1 when voxels are not found, even if they are valid
      # also, to JIT, the return shape must be predictable...
      return culled_voxels, attrs

  def set(self, values):
      v,a = self._prep_generic_setter()
      a = a.at[:].set(values)
      return self.voxlist.set_voxel(v, a)

  def apply(self, func):
      v, a = self._prep_generic_setter()
      a = a.at[:].apply(func)
      return self.voxlist.set_voxel(v, a)

  def add(self, values):
      v, a = self._prep_generic_setter()
      a = a.at[:].add(values)
      return self.voxlist.set_voxel(v, a)

  def multiply(self, values):
      v, a = self._prep_generic_setter()
      a = a.at[:].multiply(values)
      return self.voxlist.set_voxel(v, a)
  mul = multiply

  def divide(self, values):
      v, a = self._prep_generic_setter()
      a = a.at[:].divide(values)
      return self.voxlist.set_voxel(v, a)

  def power(self, values):
      v, a = self._prep_generic_setter()
      a = a.at[:].power(values)
      return self.voxlist.set_voxel(v, a)

  def min(self, values):
      v, a = self._prep_generic_setter()

      a_done = a.clip(0, jnp.inf).at[:].min(values)
      is_minus_1_mask = (a == -1).astype(jnp.float32)

      a = a * is_minus_1_mask + a_done * (1-is_minus_1_mask)
      return self.voxlist.set_voxel(v, a)

  def max(self, values):
      v, a = self._prep_generic_setter()

      a_done = a.at[:].max(values)
      is_minus_1_mask = (a == -1).astype(jnp.float32)

      a = a * is_minus_1_mask + a_done * (1 - is_minus_1_mask)
      return self.voxlist.set_voxel(v, a)
