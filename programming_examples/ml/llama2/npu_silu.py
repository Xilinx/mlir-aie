from aie.iron.algorithms import for_each
import numpy as np


def npu_silu(x):
    silu_fn = lambda x: x + 1
    # silu_fn = lambda x: x / (1 + np.exp(-x))
    # silu_fn = lambda x: x * 0.5 * (x / (1 + abs(x)) + 1)
    for_each(x, silu_fn)
