from jaxvox.slicer import concretize_index

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

  def set(self, values):
      # todo for set and all other setter funcs:
      # need to make sure all indexes are in the list first
      # then get the values
      # then apply the setter
      # then add them AGAIN, OR replace their attr value if we have the idxs
      voxels = self.index
      voxels = self.voxlist
      idxs, attrs = self.voxlist.find(voxels)

      #attrs.

      return self.voxlist.set_voxel(self.index, values)
      #new_grid = self.voxlist.grid.at[self.index].set(values=values, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode="drop")
      #return self.voxlist.replace(grid=new_grid)

  def apply(self, func):
      voxlist = self.voxlist.set_voxel(self.index)


      seek_vox_idx, seek_vox_attr = self.voxlist.find(self.index)
      new_values = seek_vox_attr.at[seek_vox_idx].apply(func)
      return self.voxlist.replace(attrs=new_values)
      #values = values.apply(func)

      #new_grid = self.voxlist.grid.at[self.index].apply(func, indices_are_sorted=indices_are_sorted,
      #                                                  unique_indices=unique_indices, mode="drop")
      #return self.voxlist.replace(grid=new_grid)

  def add(self, values):
      seek_vox_idx, seek_vox_attr = self.voxlist.find(self.index)
      new_values = seek_vox_attr.at[seek_vox_idx].add(values)
      return self.voxlist.replace(attrs=new_values)

  def multiply(self, values, *, indices_are_sorted=False, unique_indices=False,):
      new_grid = self.voxlist.grid.at[self.index].multiply(values, indices_are_sorted=indices_are_sorted,
                                                           unique_indices=unique_indices, mode="drop")

      return self.voxlist.replace(grid=new_grid)
  mul = multiply

  def divide(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxlist.grid.at[self.index].divide(values, indices_are_sorted=indices_are_sorted,
                                                         unique_indices=unique_indices, mode="drop")

      return self.voxlist.replace(grid=new_grid)

  def power(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxlist.grid.at[self.index].power(values, indices_are_sorted=indices_are_sorted,
                                                        unique_indices=unique_indices, mode="drop")

      return self.voxlist.replace(grid=new_grid)

  def min(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxlist.grid.at[self.index].min(values, indices_are_sorted=indices_are_sorted,
                                                      unique_indices=unique_indices, mode="drop")

      return self.voxlist.replace(grid=new_grid)

  def max(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxlist.grid.at[self.index].max(values, indices_are_sorted=indices_are_sorted,
                                                      unique_indices=unique_indices, mode="drop")

      return self.voxlist.replace(grid=new_grid)
