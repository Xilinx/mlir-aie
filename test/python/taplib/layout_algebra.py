# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from _legacy_tensortiler2d import TensorTiler2D
from aie.helpers.taplib import Layout, TensorAccessPattern
from aie.helpers.taplib.symbolic import sceildiv, smin, sprod
from numpy.lib.stride_tricks import as_strided
from util import construct_test

# RUN: %python %s | FileCheck %s


def visited(layout: Layout) -> np.ndarray:
    """Element indices a view visits, in order, from an as_strided oracle."""
    base = np.arange(int(np.prod(layout.tensor_dims)), dtype=np.int64)
    itemsize = base.itemsize
    view = as_strided(
        base[layout.offset :],
        shape=tuple(layout.sizes),
        strides=tuple(s * itemsize for s in layout.strides),
        writeable=False,
    )
    return view.ravel()


def tap_order(layout: Layout) -> np.ndarray:
    """Return same walk through TensorAccessPattern's generator, for cross-checking."""
    return np.fromiter(layout.tap(None).access_generator(), dtype=np.int64)


# CHECK-LABEL: full_and_permute
@construct_test
def full_and_permute():
    base = np.arange(6 * 4).reshape(6, 4)
    lay = Layout.full((6, 4))
    assert lay.sizes == [6, 4] and lay.strides == [4, 1] and lay.offset == 0
    assert (visited(lay) == base.ravel()).all()
    assert (tap_order(lay) == base.ravel()).all()

    t = lay.permute((1, 0))
    assert t.sizes == [4, 6] and t.strides == [1, 4]
    assert (visited(t) == base.T.ravel()).all()
    assert (tap_order(t) == base.T.ravel()).all()

    three = Layout.full((2, 3, 5))
    p = three.permute((2, 0, 1))
    assert (
        visited(p) == np.arange(30).reshape(2, 3, 5).transpose(2, 0, 1).ravel()
    ).all()


# CHECK-LABEL: split_merge_coalesce
@construct_test
def split_merge_coalesce():
    lay = Layout.full((8, 6))
    s = lay.split(0, 2)
    assert s.sizes == [4, 2, 6] and s.strides == [12, 6, 1]
    assert (visited(s) == visited(lay)).all()
    assert s.merge(0) == lay
    # coalesce() merges maximally: a row-major tensor is one contiguous run.
    assert s.coalesce() == Layout((8, 6), 0, [48], [1])
    assert (visited(s.coalesce()) == visited(lay)).all()
    try:
        lay.split(1, 4)
        assert False
    except ValueError:
        pass
    # A column slice is not contiguous, so it must not merge.
    col = lay[:, 0:3]
    try:
        col.merge(0)
        assert False
    except ValueError:
        pass
    # Unit dims disappear; a lone unit dim survives.
    u = Layout((4,), 0, [1, 4, 1], [0, 1, 0])
    assert u.drop_unit_dims().sizes == [4]
    assert Layout((4,), 2, [1], [0]).drop_unit_dims().sizes == [1]


# CHECK-LABEL: slicing
@construct_test
def slicing():
    base = np.arange(7 * 9).reshape(7, 9)
    lay = Layout.full((7, 9))
    for key in (
        np.s_[1:5, 2:8],
        np.s_[::2, 1::3],
        np.s_[3],
        np.s_[:, -2],
        np.s_[-3:, ...],
        np.s_[None, 2:4, :],
        np.s_[..., 4],
        np.s_[2:3, 4:5],
    ):
        got = visited(lay[key])
        want = np.asarray(base[key]).ravel()
        assert got.shape == want.shape and (got == want).all(), key
    # Same numbers as TensorAccessPattern.from_slice, which this generalises.
    for key in (np.s_[1:5, 2:8], np.s_[::2, 1::3], np.s_[3]):
        assert lay[key].tap(None) == TensorAccessPattern.from_slice((7, 9), key)
    for bad in (np.s_[5:5], np.s_[::-1], np.s_[9]):
        try:
            lay[bad]
            assert False, bad
        except (ValueError, IndexError):
            pass


# CHECK-LABEL: tile_grid_indexing
@construct_test
def tile_grid_indexing():
    m, n, r, t = 8, 12, 2, 3
    base = np.arange(m * n).reshape(m, n)
    grid = Layout.full((m, n)).tile((r, t))
    assert grid.grid_shape == [m // r, n // t] and grid.tile_shape == [r, t]
    assert grid.num_steps == 16 and len(grid) == 16
    for i in range(m // r):
        for j in range(n // t):
            want = base[i * r : (i + 1) * r, j * t : (j + 1) * t].ravel()
            assert (visited(grid[i, j]) == want).all()
            assert (visited(grid.at(i, j)) == want).all()
            assert grid[i * (n // t) + j] == grid[i, j]
            assert grid.order("col")[j * (m // r) + i] == grid[i, j]
    # Iteration and materialize agree with tile_at.
    assert [lay for lay in grid] == [grid.tile_at(s) for s in range(16)]
    assert len(grid.materialize()) == 16
    # Inside-tile transpose.
    ct = grid.permute_tile((1, 0))
    assert (visited(ct[1, 2]) == base[2:4, 6:9].T.ravel()).all()
    # Rank 3 (a NHWC-style tensor) tiles the same way.
    vol = np.arange(4 * 6 * 8).reshape(4, 6, 8)
    g3 = Layout.full((4, 6, 8)).tile((2, 3, 4))
    assert (visited(g3[1, 0, 1]) == vol[2:4, 0:3, 4:8].ravel()).all()


# CHECK-LABEL: group_semantics
@construct_test
def group_semantics():
    # 16 tiles per row, repeat 2 spaced 4 apart: group j covers tiles j, j+4.
    m, n, r, t = 4, 32, 2, 2
    base = np.arange(m * n).reshape(m, n)
    grid = Layout.full((m, n)).tile((r, t)).group((1, 2), steps=(1, 4))
    assert grid.grid_shape == [2, 8]
    assert grid.tile_shape == [1, 2, r, t]
    assert grid.num_steps == 16
    tiles_per_row = n // t
    for step in range(16):
        b, j = divmod(step % 8, 4)
        row = step // 8
        first = b * 8 + j
        want = np.concatenate(
            [
                base[row * r : (row + 1) * r, tile * t : (tile + 1) * t].ravel()
                for tile in (first, first + 4)
            ]
        )
        assert (visited(grid[step]) == want).all(), step
        assert first + 4 < tiles_per_row
    # Column-major repeats swap which repeat dimension is outermost.
    g2 = Layout.full((8, 8)).tile((2, 2)).group((2, 2))
    g2c = Layout.full((8, 8)).tile((2, 2)).group((2, 2), col_major=True)
    assert g2.tile_shape == [2, 2, 2, 2] and g2.tile_strides == [16, 2, 8, 1]
    assert g2c.tile_strides == [2, 16, 8, 1]
    # Row/col step order survives grouping.
    g2r = Layout.full((8, 16)).tile((2, 2)).group((2, 2))
    assert (
        g2r.order("col")[1] == g2r[4]
    )  # second step walks down the first column of groups


# CHECK-LABEL: repeat_and_tap_form
@construct_test
def repeat_and_tap_form():
    lay = Layout.full((1, 16)).repeat(3)
    assert lay.sizes == [3, 1, 16] and lay.strides == [0, 16, 1]
    assert (visited(lay) == np.tile(np.arange(16), 3)).all()
    # The shim form keeps a pure repeat in slot 0 (the queue repeat) and pads
    # after it; a real dimension pads on the left.
    assert lay.tap() == TensorAccessPattern((1, 16), 0, [3, 1, 1, 16], [0, 0, 0, 1])
    assert Layout.full((4, 8)).tap() == TensorAccessPattern(
        (4, 8), 0, [1, 1, 4, 8], [0, 0, 8, 1]
    )
    # A grid tile with a repeat: [R, th, tw] -> [R, 1, th, tw].
    g = Layout.full((8, 8)).tile((2, 8)).repeat(3)
    assert g[1].tap() == TensorAccessPattern((8, 8), 16, [3, 1, 2, 8], [0, 0, 8, 1])
    # Too many dims is an error, not a silent drop.
    try:
        Layout.full((2, 2, 2, 2, 2)).tap()
        assert False
    except ValueError:
        pass
    assert Layout.full((2, 2, 2, 2, 2)).tap(None).sizes == [2, 2, 2, 2, 2]


# CHECK-LABEL: matmul_stream_dims
@construct_test
def matmul_stream_dims():
    m, k, n, r, s, t = 64, 64, 64, 4, 8, 4
    # A operand: what TensorTiler2D.group_tiler((m,k),(r,s),(m//r,k//s))[0] gives today.
    a = Layout.full((m, k)).tile((r, s)).layout.stream_dims()
    assert a == [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    legacy = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    assert a == list(legacy.transformation_dims)
    # All four operand layouts kernels/linalg.py derives, including the
    # transposed-B and column-major-C variants.
    assert Layout.full((n, k)).tile((t, s)).layout.stream_dims() == [
        (n // t, t * k),
        (k // s, s),
        (t, k),
        (s, 1),
    ]
    assert Layout.full((n, m)).tile((t, r)).inverse().stream_dims() == [
        (n // t, t * m),
        (t, r),
        (m // r, r * t),
        (r, 1),
    ]
    # C operand: the "un-blocking" order that linalg.py hand-writes today.
    c = Layout.full((m, n)).tile((r, t)).inverse()
    assert c.stream_dims() == [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    # Reading a tile-blocked buffer with that view yields logical row-major order.
    logical = np.arange(m * n).reshape(m, n)
    blocked = logical.reshape(m // r, r, n // t, t).transpose(0, 2, 1, 3).ravel()
    assert (blocked[visited(c)] == logical.ravel()).all()
    # And it is the inverse of tiling: tile the row-major tensor, concatenate
    # the tiles, read back with inverse().
    tiles = Layout.full((m, n)).tile((r, t))
    cat = np.concatenate([visited(tiles[i]) for i in range(len(tiles))])
    assert (cat[visited(c)] == np.arange(m * n)).all()
    # Step order and the walk inside each tile are part of the blocked buffer.
    for grid in (
        Layout.full((8, 12)).tile((2, 3)).order("col"),
        Layout.full((8, 12)).tile((2, 3)).permute_tile((1, 0)),
        Layout.full((8, 12)).tile((2, 3)).order("col").permute_tile((1, 0)),
        Layout.full((4, 6, 8)).tile((2, 3, 4)).order((2, 0, 1)).permute_tile((1, 2, 0)),
    ):
        cat = np.concatenate([visited(grid[i]) for i in range(len(grid))])
        size = int(np.prod(grid.layout.tensor_dims))
        assert (cat[visited(grid.inverse())] == np.arange(size)).all()


# CHECK-LABEL: partition_and_partial
@construct_test
def partition_and_partial():
    # The per-column chunk idiom: [1, 1, 1, N // k] at offset i * N // k.
    N, k = 4096, 8
    parts = Layout.full((1, N)).partition(k)
    assert len(parts) == k
    for i in range(k):
        assert parts[i].tap() == TensorAccessPattern(
            (1, N), i * (N // k), [1, 1, 1, N // k], [0, 0, 0, 1]
        )
    # Partition along a leading axis keeps the rows contiguous.
    rows = Layout.full((64, 32)).partition(4, dim=0)
    assert rows[1].tap() == TensorAccessPattern(
        (64, 32), 512, [1, 1, 16, 32], [0, 0, 32, 1]
    )
    # A ragged group: 14 tiles, repeat 7 spaced 3 apart -> repeat capped at 5,
    # then 5, 5, 4 tiles for the three groups.
    g = Layout.full((3, 28)).tile((3, 2)).group((1, 7), steps=(1, 3), partial=True)
    assert g.grid_shape == [1, 3]
    assert [g[i].sizes[0] for i in range(3)] == [5, 5, 4]
    base = np.arange(3 * 28).reshape(3, 28)
    for j in range(3):
        tiles = list(range(j, 14, 3))[: g[j].sizes[0]]
        want = np.concatenate([base[:, t * 2 : (t + 1) * 2].ravel() for t in tiles])
        assert (visited(g[j]) == want).all()
    try:
        Layout.full((3, 28)).tile((3, 2)).group((1, 7), steps=(1, 3))
        assert False
    except ValueError as e:
        assert "partial=True" in str(e)


# CHECK-LABEL: symbolic_helpers_on_ints
@construct_test
def symbolic_helpers_on_ints():
    assert smin(3, 5) == 3 and smin(np.int32(7), 2) == 2
    assert sceildiv(7, 3) == 3 and sceildiv(6, 3) == 2 and sceildiv(0, 4) == 0
    assert sprod([2, 3, np.int64(4)]) == 24 and isinstance(sprod([2, 3]), int)
    try:
        Layout.full((8, 8)).split(0, 3)
        assert False
    except ValueError as e:
        assert "not divisible" in str(e)
