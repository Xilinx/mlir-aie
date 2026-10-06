# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from util import accesses, construct_test, grid_steps

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: every_grouping_partitions_the_tensor
@construct_test
def every_grouping_partitions_the_tensor():
    """Sweep step groups over steps, repeats, column-major and ragged edges.

    Whatever the grouping, the steps of a grid must visit every element of
    the tensor exactly once between them: no gap, no overlap, and a ragged
    last group holding only the tiles that remain.
    """
    checked = 0
    for rows, cols in ((1, 12), (6, 8), (7, 5)):
        for tr, tc in ((1, 1), (1, 2), (3, 1)):
            if rows % tr or cols % tc:
                continue
            grid = TensorAccessPattern.full((rows, cols)).tile((tr, tc))
            for s0, s1, r0, r1 in itertools.product(
                (1, 2, 3), (1, 2, 4), (1, 2, 3), (1, 2)
            ):
                for order in ("row", "col"):
                    for partial in (False, True):
                        try:
                            g = grid_steps(
                                grid,
                                (r0, r1),
                                steps=(s0, s1),
                                order=order,
                                partial=partial,
                            )
                        except ValueError as e:
                            assert not partial and "not divisible" in str(e), e
                            continue
                        assert (accesses(g)[1] == 1).all(), (
                            (rows, cols),
                            (tr, tc),
                            (s0, s1),
                            (r0, r1),
                            order,
                            partial,
                        )
                        checked += 1
    print(f"groupings checked: {checked}, every one a partition")
    # CHECK: groupings checked: 912, every one a partition


# CHECK-LABEL: inverse_round_trips
@construct_test
def inverse_round_trips():
    """Store a grid's tiles one after another, then read them back with inverse().

    The blocked buffer must come back in logical row-major order, whatever the
    step order or in-tile order. A tiling of a sub-view is not a permutation of
    its tensor, so it has no inverse; its blocked buffer is the full view's.
    """
    for rows, cols, tr, tc in ((8, 8, 2, 4), (12, 6, 3, 2), (16, 32, 4, 8)):
        host = np.arange(rows * 2 * cols).reshape(rows, 2 * cols)
        full = TensorAccessPattern.full((rows, cols)).tile((tr, tc))
        sub = TensorAccessPattern.full((rows, 2 * cols))[:, cols:].tile((tr, tc))
        for grid in (
            full,
            full.permute((1, 0, 2, 3)),
            full.permute((0, 1, 3, 2)),
            full.permute((1, 0, 3, 2)),
        ):
            src = host[:, :cols]
            blocked = np.concatenate([tile.gather(src) for row in grid for tile in row])
            view = grid.inverse().gather(blocked).reshape(rows, cols)
            assert (view == src).all()
        blocked = np.concatenate([tile.gather(host) for row in sub for tile in row])
        view = full.inverse().gather(blocked).reshape(rows, cols)
        assert (view == host[:, cols:]).all()
        try:
            sub.inverse()
            assert False
        except ValueError:
            pass
    print("inverse round trips")
    # CHECK: inverse round trips


# CHECK-LABEL: slices_match_numpy
@construct_test
def slices_match_numpy():
    a = np.arange(6 * 8).reshape(6, 8)
    full = TensorAccessPattern.full((6, 8))
    for key in (
        slice(1, 5),
        (slice(None), slice(2, 7)),
        (slice(0, 6, 2), slice(1, 8, 3)),
        (3, slice(None)),
        (slice(None), 4),
    ):
        assert (full[key].gather(a) == a[key].reshape(-1)).all(), key
    assert (full.T.gather(a) == a.T.reshape(-1)).all()
    assert full.split(1, 4).merge(1).gather(a).tolist() == list(range(48))
    print("slices, permute, split/merge match numpy")
    # CHECK: slices, permute, split/merge match numpy
