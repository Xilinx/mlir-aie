import collections
from itertools import islice, zip_longest
import numbers

from aie.extras.dialects.ext.memref import MemRef
import numpy as np
from numpy.lib.stride_tricks import as_strided

from aie.dialects import aie
from aie.extras.util import find_ops
from aie.ir import UnitAttr


def grouper(iterable, n, *, incomplete="fill", fill_value=None):
    args = [iter(iterable)] * n
    match incomplete:
        case "fill":
            return zip_longest(*args, fillvalue=fill_value)
        case "strict":
            return zip(*args, strict=True)
        case "ignore":
            return zip(*args)
        case _:
            raise ValueError("Expected fill, strict, or ignore")


def sliding_window(iterable, n):
    it = iter(iterable)
    window = collections.deque(islice(it, n - 1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)


def display_flows(module):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()
    for c in find_ops(
        module.operation,
        lambda o: isinstance(o.operation.opview, aie.FlowOp),
    ):
        arrow = mpatches.FancyArrowPatch(
            (c.source.owner.opview.col.value, c.source.owner.opview.row.value),
            (c.dest.owner.opview.col.value, c.dest.owner.opview.row.value),
            mutation_scale=10,
        )
        axs.add_patch(arrow)

    axs.set(xlim=(-1, 5), ylim=(-1, 6))
    fig.show()
    fig.tight_layout()
    fig.savefig("flows.png")


def annot(op, annot):
    op.operation.attributes[annot] = UnitAttr.get()


def extract_patches(
    arr=None,
    arr_shape=None,
    patch_shape: int | tuple[int, ...] | list[int, ...] = 8,
    extraction_step: int | tuple[int, ...] | list[int, ...] = None,
    dtype: np.dtype = None,
    trailing_dims=4,
    transpose=False,
):
    if dtype is None:
        dtype = np.int32()
    if isinstance(arr, MemRef):
        arr_shape = arr.shape
        arr = None
    if arr is None:
        arr = np.empty(arr_shape, dtype=dtype)
    if transpose:
        arr = arr.T

    if extraction_step is None:
        extraction_step = patch_shape
    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    # grab the elements at the starts of the "extraction steps"
    # and get the strides to those elements
    indexing_strides = arr[slices].strides

    patch_indices_shape = (
        (np.array(arr.shape) - np.array(patch_shape)) // np.array(extraction_step)
    ) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = list(indexing_strides) + list(patch_strides)
    if transpose:
        strides[-2], strides[-1] = strides[-1], strides[-2]

    patches = as_strided(arr, shape=shape, strides=strides)
    if arr_shape is not None:
        return list(zip(patches.shape, np.array(patches.strides) // dtype.itemsize))[
            -trailing_dims:
        ]
    else:
        return patches
