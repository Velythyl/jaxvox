import functools

import jax.numpy as jnp
import jax

def _slice_tree_flatten(self):
    return tuple(), (self.start, self.step, self.stop)

def _slice_tree_unflatten(aux_data, children):
    return slice(aux_data[0], aux_data[1], aux_data[2])


jax.tree_util.register_pytree_node(slice,
                                   _slice_tree_flatten,
                                   _slice_tree_unflatten)

@functools.partial(jax.jit, static_argnums=(0,))
def concretize_index(shape, slices):
    all_indices_1d = [jnp.arange(shape_i) for shape_i in shape]

    NUM_DIMS = len(shape)
    NUM_SLICES = len(slices)
    MISSING_SLICES = NUM_DIMS - NUM_SLICES

    slices = list(slices) + [None] * MISSING_SLICES

    filtered_indices_1d = []
    for i, (slice_i, i_indices_1d) in enumerate(zip(slices, all_indices_1d)):
        filtered_indices_1d.append(
            i_indices_1d[slice_i].squeeze()
        )

    return jnp.stack(jnp.meshgrid(*filtered_indices_1d, indexing="ij"), axis=-1).reshape(-1,NUM_DIMS)


if __name__ == "__main__":
    shape = (4, 5, 6)
    slices = (slice(1, 3), slice(2, 4))  # Select rows 1-2, cols 2-3
    #print(jnp.ogrid[slices])
    #

   # temp = jnp.indices(shape, sparse=True)

    #sparse_indices = jnp.indices(shape, sparse=True)

    # Apply the slices to each dimension
    #selected_indices = [sparse_indices[i][ranges[i]] for i in range(len(arr_shape))]

    # Use broadcasting to combine the sparse indices
    #indices = np.stack(np.meshgrid(*selected_indices, indexing='ij'), axis=-1)


    indices = concretize_index_3d(shape, slices)
    print(indices)
    # Output: [[1 2]
    #          [1 3]
    #          [2 2]
    #          [2 3]]