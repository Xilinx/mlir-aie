# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

import numpy as np
from aie.helpers.taplib import Layout
from util import construct_test

# RUN: %python %s | FileCheck %s


def indices(lay):
    idx = np.zeros((), dtype=np.int64) + lay.offset
    for size, stride in zip(lay.sizes, lay.strides):
        idx = idx[..., None] + np.arange(size, dtype=np.int64) * stride
    return idx.reshape(-1)


# CHECK-LABEL: every_grouping_partitions_the_tensor
@construct_test
def every_grouping_partitions_the_tensor():
    """Sweep group() over steps, repeats, column-major and ragged edges.

    Whatever the grouping, the steps of a grid must visit every element of
    the tensor exactly once between them: no gap, no overlap, and a ragged
    last group holding only the tiles that remain.
    """
    checked = 0
    for rows, cols in ((1, 12), (6, 8), (7, 5)):
        for tr, tc in ((1, 1), (1, 2), (3, 1)):
            if rows % tr or cols % tc:
                continue
            grid = Layout.full((rows, cols)).tile((tr, tc))
            for s0, s1, r0, r1 in itertools.product(
                (1, 2, 3), (1, 2, 4), (1, 2, 3), (1, 2)
            ):
                for col_major in (False, True):
                    for partial in (False, True):
                        try:
                            g = grid.group(
                                (r0, r1),
                                steps=(s0, s1),
                                col_major=col_major,
                                partial=partial,
                            )
                        except ValueError as e:
                            assert not partial and "not divisible" in str(e), e
                            continue
                        seen = np.zeros(rows * cols, dtype=int)
                        for k in range(g.num_steps):
                            seen[indices(g[k])] += 1
                        assert (seen == 1).all(), (
                            (rows, cols),
                            (tr, tc),
                            (s0, s1),
                            (r0, r1),
                            col_major,
                            partial,
                        )
                        checked += 1
    print(f"groupings checked: {checked}, every one a partition")
    # CHECK: groupings checked: 912, every one a partition


# CHECK-LABEL: inverse_round_trips
@construct_test
def inverse_round_trips():
    """tile() then inverse(): reading the blocked buffer back is the identity."""
    for rows, cols, tr, tc in ((8, 8, 2, 4), (12, 6, 3, 2), (16, 32, 4, 8)):
        grid = Layout.full((rows, cols)).tile((tr, tc))
        blocked = indices(grid.layout)
        assert (blocked[indices(grid.inverse())] == np.arange(rows * cols)).all()
    print("inverse round trips")
    # CHECK: inverse round trips


# CHECK-LABEL: slices_match_numpy
@construct_test
def slices_match_numpy():
    a = np.arange(6 * 8).reshape(6, 8)
    for key in (
        slice(1, 5),
        (slice(None), slice(2, 7)),
        (slice(0, 6, 2), slice(1, 8, 3)),
        (3, slice(None)),
        (slice(None), 4),
    ):
        assert (indices(Layout.full((6, 8))[key]) == a[key].reshape(-1)).all(), key
    assert (indices(Layout.full((6, 8)).permute((1, 0))) == a.T.reshape(-1)).all()
    assert indices(Layout.full((6, 8)).split(1, 4).merge(1)).tolist() == list(range(48))
    print("slices, permute, split/merge match numpy")
    # CHECK: slices, permute, split/merge match numpy
