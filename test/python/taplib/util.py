# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

import numpy as np


# Run test
def construct_test(f):
    print("\nTEST:", f.__name__)
    f()


def grid_steps(
    tiles,
    repeats=(1, 1),
    steps=(1, 1),
    order="row",
    group_order="row",
    partial=False,
    repeat=1,
):
    """Split a 2-D tiling into the per-step taps of TensorTiler2D.step_tiler.

    Along grid axis i, a block is steps[i] * repeats[i] tiles and group j of a
    block takes tiles j, j + steps[i], ... ; each step is one group, a slice
    of the tiling. order is the step order over the grid axes, group_order the
    walk order over a group's tiles, and repeat walks each step that many
    times. With partial, the groups at the tensor edge may be short.
    """
    axes = []
    for g, s, r in zip(tiles.sizes[:2], steps, repeats):
        s = 1 if s > g else s
        if partial:
            r = min(r, -(-g // s))
        elif g % (s * r):
            raise ValueError(f"{g} tiles are not divisible into groups of {s * r}")
        positions = []
        for block in range(-(-g // (s * r))):
            for j in range(s):
                first = block * s * r + j
                if first < g:
                    count = min(r, -(-(g - first) // s))
                    positions.append(
                        first if count == 1 else slice(first, first + count * s, s)
                    )
        axes.append(positions)
    if order == "col":
        pairs = [(i, j) for j, i in itertools.product(axes[1], axes[0])]
    else:
        pairs = list(itertools.product(axes[0], axes[1]))
    taps = []
    for i, j in pairs:
        t = tiles[i, j]
        if group_order == "col" and isinstance(i, slice) and isinstance(j, slice):
            t = t.permute((1, 0, 2, 3))
        if repeat != 1:
            t = t.repeat(repeat)
        taps.append(t)
    return taps


def accesses(taps):
    """Return the access_order and access_count of taps walked one after another."""
    order = np.full(taps[0].tensor_dims, -1)
    count = np.zeros(taps[0].tensor_dims, dtype=int)
    walked = 0
    for t in taps:
        t_order, t_count = t.accesses()
        order = np.where(t_order >= 0, t_order + walked, order)
        count += t_count
        walked += t.numel
    return order, count
