from mem_func_class import (
    membershipFunctions,
    gaussian,
    gbell,
    trapezoidal,
    triangular,
    sigmoid,
)
import jax.numpy as jnp

X = jnp.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]
)

mfs_default = membershipFunctions(X, type="trapezoidal", num=3)  # for all the same
print(mfs_default)