class _VoxGridIndexUpdateHelper:

    __slots__ = ("voxgrid",)
    def __init__(self, voxgrid):
        self.voxgrid = voxgrid

    def __getitem__(self, index):
        return _IndexUpdateRef(self.voxgrid, index)

    def __repr__(self):
        return f"_IndexUpdateHelper({repr(self.voxgrid)})"


class _IndexUpdateRef:
  __slots__ = ("voxgrid", "index")

  def __init__(self, voxgrid, index):
    self.voxgrid = voxgrid
    self.index = index

  def __repr__(self):
    return f"_IndexUpdateRef({repr(self.voxgrid)}, {repr(self.index)})"

  def get(self, indices_are_sorted=False, unique_indices=False):
      return self.voxgrid.grid.at[self.index].get(indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, fill_value=-1, mode="fill")

  def set(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].set(values=values, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode="drop")
      return self.voxgrid.replace(grid=new_grid)

  def apply(self, func, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].apply(func, indices_are_sorted=indices_are_sorted,
                                                        unique_indices=unique_indices, mode="drop")
      return self.voxgrid.replace(grid=new_grid)

  def add(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].add(values, indices_are_sorted=indices_are_sorted, unique_indices=unique_indices, mode="drop")
      return self.voxgrid.replace(grid=new_grid)

  def multiply(self, values, *, indices_are_sorted=False, unique_indices=False,):
      new_grid = self.voxgrid.grid.at[self.index].multiply(values, indices_are_sorted=indices_are_sorted,
                                                           unique_indices=unique_indices, mode="drop")

      return self.voxgrid.replace(grid=new_grid)
  mul = multiply

  def divide(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].divide(values, indices_are_sorted=indices_are_sorted,
                                                         unique_indices=unique_indices, mode="drop")

      return self.voxgrid.replace(grid=new_grid)

  def power(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].power(values, indices_are_sorted=indices_are_sorted,
                                                        unique_indices=unique_indices, mode="drop")

      return self.voxgrid.replace(grid=new_grid)

  def min(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].min(values, indices_are_sorted=indices_are_sorted,
                                                      unique_indices=unique_indices, mode="drop")

      return self.voxgrid.replace(grid=new_grid)

  def max(self, values, *, indices_are_sorted=False, unique_indices=False):
      new_grid = self.voxgrid.grid.at[self.index].max(values, indices_are_sorted=indices_are_sorted,
                                                      unique_indices=unique_indices, mode="drop")

      return self.voxgrid.replace(grid=new_grid)
