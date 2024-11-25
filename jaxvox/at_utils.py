import struct
from typing import Union

import jaxvox._jaxvox as jaxvox

Type_Concrete_VoxCol = Union[jaxvox.VoxList, jaxvox.VoxGrid]

#@struct.dataclass

class _IndexUpdateHelper:
    vox_grid_or_list: Type_Concrete_VoxCol

    __slots__ = ("vox_grid_or_list",)

    def __init__(self, vox_grid_or_list):
        self.vox_grid_or_list = vox_grid_or_list

    def __getitem__(self, index):
        return _IndexUpdateRef(self.array, index)

    def __repr__(self):
        return f"_IndexUpdateHelper({repr(self.vox_grid_or_list)})"


class _IndexUpdateRef:
  __slots__ = ("vox_grid_or_list", "index")

  def __init__(self, vox_grid_or_list: Type_Concrete_VoxCol, index):
    self.vox_grid_or_list = vox_grid_or_list
    self.index = index

  def __repr__(self):
    return f"_IndexUpdateRef({repr(self.vox_grid_or_list)}, {repr(self.index)})"

  def get(self, *, indices_are_sorted=False, unique_indices=False,
          mode=None, fill_value=None):
    """Equivalent to ``x[idx]``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexing <numpy.doc.indexing>` ``x[idx]``. This function differs from
    the usual array indexing syntax in that it allows additional keyword
    arguments ``indices_are_sorted`` and ``unique_indices`` to be passed.

    See :mod:`jax.ops` for details.
    """
    return lax_numpy._rewriting_take(self.array, self.index,
                                     indices_are_sorted=indices_are_sorted,
                                     unique_indices=unique_indices, mode=mode,
                                     fill_value=fill_value)

  @allow_pass_by_position_with_warning
  def set(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:`indexed assignment <numpy.doc.indexing>` ``x[idx] = y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values, lax.scatter,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  @allow_pass_by_position_with_warning
  def apply(self, func, *, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``func.at(x, idx)`` for a unary ufunc ``func``.

    Returns the value of ``x`` that would result from applying the unary
    function ``func`` to ``x`` at the given indices. This is similar to
    ``x.at[idx].set(func(x[idx]))``, but differs in the case of repeated indices:
    in ``x.at[idx].apply(func)``, repeated indices result in the function being
    applied multiple times.

    Note that in the current implementation, ``scatter_apply`` is not compatible
    with automatic differentiation.

    See :mod:`jax.ops` for details.
    """
    def _scatter_apply(x, indices, y, dims, **kwargs):
      return lax.scatter_apply(x, indices, func, dims, update_shape=y.shape, **kwargs)
    return scatter._scatter_update(self.array, self.index,
                                   lax_internal._zero(self.array.dtype),
                                   _scatter_apply,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  @allow_pass_by_position_with_warning
  def add(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] += y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] += y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_add,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  @allow_pass_by_position_with_warning
  def multiply(self, values, *, indices_are_sorted=False, unique_indices=False,
               mode=None):
    """Pure equivalent of ``x[idx] *= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] *= y``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_mul,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices,
                                   mode=mode)
  mul = multiply

  @allow_pass_by_position_with_warning
  def divide(self, values, *, indices_are_sorted=False, unique_indices=False,
             mode=None):
    """Pure equivalent of ``x[idx] /= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] /= y``.

    See :mod:`jax.ops` for details.
    """
    return ufuncs.divide(
      self.array,
      scatter._scatter_update(lax_numpy.ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  @allow_pass_by_position_with_warning
  def power(self, values, *, indices_are_sorted=False, unique_indices=False,
            mode=None):
    """Pure equivalent of ``x[idx] **= y``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>` ``x[idx] **= y``.

    See :mod:`jax.ops` for details.
    """
    return ufuncs.power(
      self.array,
      scatter._scatter_update(lax_numpy.ones_like(self.array), self.index, values,
                              lax.scatter_mul,
                              indices_are_sorted=indices_are_sorted,
                              unique_indices=unique_indices, mode=mode))

  @allow_pass_by_position_with_warning
  def min(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = minimum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = minimum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_min,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)

  @allow_pass_by_position_with_warning
  def max(self, values, *, indices_are_sorted=False, unique_indices=False,
          mode=None):
    """Pure equivalent of ``x[idx] = maximum(x[idx], y)``.

    Returns the value of ``x`` that would result from the NumPy-style
    :mod:indexed assignment <numpy.doc.indexing>`
    ``x[idx] = maximum(x[idx], y)``.

    See :mod:`jax.ops` for details.
    """
    return scatter._scatter_update(self.array, self.index, values,
                                   lax.scatter_max,
                                   indices_are_sorted=indices_are_sorted,
                                   unique_indices=unique_indices, mode=mode)