# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from Inputs.legacy_tensortiler2d import TensorTiler2D
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.taplib.symbolic import sceildiv, smin, sprod
from numpy.lib.stride_tricks import as_strided
from util import construct_test

# RUN: %python %s | FileCheck %s


def visited(tap: TensorAccessPattern) -> np.ndarray:
    """Element indices a pattern visits, in order, from an as_strided oracle."""
    base = np.arange(int(np.prod(tap.tensor_dims)), dtype=np.int64)
    itemsize = base.itemsize
    view = as_strided(
        base[tap.offset :],
        shape=tuple(tap.sizes),
        strides=tuple(s * itemsize for s in tap.strides),
        writeable=False,
    )
    return view.ravel()


def tap_order(tap: TensorAccessPattern) -> np.ndarray:
    """Return same walk through TensorAccessPattern's generator, for cross-checking."""
    return np.fromiter(tap.access_generator(), dtype=np.int64)


# CHECK-LABEL: full_and_permute
@construct_test
def full_and_permute():
    base = np.arange(6 * 4).reshape(6, 4)
    lay = TensorAccessPattern.full((6, 4))
    assert lay.sizes == [6, 4] and lay.strides == [4, 1] and lay.offset == 0
    assert (visited(lay) == base.ravel()).all()
    assert (tap_order(lay) == base.ravel()).all()

    t = lay.permute((1, 0))
    assert t.sizes == [4, 6] and t.strides == [1, 4]
    assert (visited(t) == base.T.ravel()).all()
    assert (tap_order(t) == base.T.ravel()).all()

    three = TensorAccessPattern.full((2, 3, 5))
    p = three.permute((2, 0, 1))
    assert (
        visited(p) == np.arange(30).reshape(2, 3, 5).transpose(2, 0, 1).ravel()
    ).all()


# CHECK-LABEL: split_merge_coalesce
@construct_test
def split_merge_coalesce():
    lay = TensorAccessPattern.full((8, 6))
    s = lay.split(0, 2)
    assert s.sizes == [4, 2, 6] and s.strides == [12, 6, 1]
    assert (visited(s) == visited(lay)).all()
    assert s.merge(0) == lay
    # coalesce() merges maximally: a row-major tensor is one contiguous run.
    assert s.coalesce() == TensorAccessPattern((8, 6), 0, [48], [1])
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
    u = TensorAccessPattern((4,), 0, [1, 4, 1], [0, 1, 0])
    assert u.coalesce().sizes == [4]
    assert TensorAccessPattern((4,), 2, [1], [0]).coalesce().sizes == [1]
    # Unit dims never step, so == ignores them.
    assert u == TensorAccessPattern((4,), 0, [4], [1])
    assert hash(u) == hash(TensorAccessPattern((4,), 0, [4], [1]))


# CHECK-LABEL: slicing
@construct_test
def slicing():
    base = np.arange(7 * 9).reshape(7, 9)
    lay = TensorAccessPattern.full((7, 9))
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
    assert lay[1:5, 2:8] == TensorAccessPattern((7, 9), 11, [4, 6], [9, 1])
    assert lay[::2, 1::3] == TensorAccessPattern((7, 9), 1, [4, 3], [18, 3])
    assert lay[3] == TensorAccessPattern((7, 9), 27, [9], [1])
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
    grid = TensorAccessPattern.full((m, n)).tile((r, t))
    assert grid.grid_shape == [m // r, n // t] and grid.tile_shape == [r, t]
    assert grid.num_steps == 16 and len(grid) == 16
    for i in range(m // r):
        for j in range(n // t):
            want = base[i * r : (i + 1) * r, j * t : (j + 1) * t].ravel()
            assert (visited(grid[i, j]) == want).all()
            assert grid[i * (n // t) + j] == grid[i, j]
            assert grid.order("col")[j * (m // r) + i] == grid[i, j]
    assert list(grid) == [grid[s] for s in range(16)]
    assert len(grid) == 16
    # Inside-tile transpose.
    ct = grid.permute_tile((1, 0))
    assert (visited(ct[1, 2]) == base[2:4, 6:9].T.ravel()).all()
    # Rank 3 (a NHWC-style tensor) tiles the same way.
    vol = np.arange(4 * 6 * 8).reshape(4, 6, 8)
    g3 = TensorAccessPattern.full((4, 6, 8)).tile((2, 3, 4))
    assert (visited(g3[1, 0, 1]) == vol[2:4, 0:3, 4:8].ravel()).all()


# CHECK-LABEL: group_semantics
@construct_test
def group_semantics():
    # 16 tiles per row, repeat 2 spaced 4 apart: group j covers tiles j, j+4.
    m, n, r, t = 4, 32, 2, 2
    base = np.arange(m * n).reshape(m, n)
    grid = TensorAccessPattern.full((m, n)).tile((r, t)).group((1, 2), steps=(1, 4))
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
    g2 = TensorAccessPattern.full((8, 8)).tile((2, 2)).group((2, 2))
    g2c = TensorAccessPattern.full((8, 8)).tile((2, 2)).group((2, 2), order="col")
    assert g2.tile_shape == [2, 2, 2, 2] and g2.tile_strides == [16, 2, 8, 1]
    assert g2c.tile_strides == [2, 16, 8, 1]
    # Row/col step order survives grouping.
    g2r = TensorAccessPattern.full((8, 16)).tile((2, 2)).group((2, 2))
    assert (
        g2r.order("col")[1] == g2r[4]
    )  # second step walks down the first column of groups


# CHECK-LABEL: repeat_and_dma_form
@construct_test
def repeat_and_dma_form():
    lay = TensorAccessPattern.full((1, 16)).repeat(3)
    assert lay.sizes == [3, 1, 16] and lay.strides == [0, 0, 1]
    assert (visited(lay) == np.tile(np.arange(16), 3)).all()
    # The shim form keeps a pure repeat in slot 0 (the queue repeat) and pads
    # after it; a real dimension pads on the left.
    shim = lay._dma_form()
    assert shim.sizes == [3, 1, 1, 16] and shim.strides == [0, 0, 0, 1]
    shim = TensorAccessPattern.full((4, 8))._dma_form()
    assert shim.sizes == [1, 1, 4, 8] and shim.strides == [0, 0, 8, 1]
    # A grid tile with a repeat: [R, th, tw] -> [R, 1, th, tw].
    g = TensorAccessPattern.full((8, 8)).tile((2, 8)).repeat(3)
    shim = g[1]._dma_form()
    assert shim.sizes == [3, 1, 2, 8] and shim.strides == [0, 0, 8, 1]
    assert shim == g[1] and shim.offset == 16
    # Too many dims for a shim DMA is an error, not a silent drop.
    try:
        TensorAccessPattern.full((2, 2, 2, 2, 2))._dma_form()
        assert False
    except ValueError:
        pass
    assert TensorAccessPattern.full((2, 2, 2, 2, 2)).sizes == [2, 2, 2, 2, 2]


# CHECK-LABEL: matmul_transformation_dims
@construct_test
def matmul_transformation_dims():
    m, k, n, r, s, t = 64, 64, 64, 4, 8, 4
    # A operand: what TensorTiler2D.group_tiler((m,k),(r,s),(m//r,k//s))[0] gave.
    a = TensorAccessPattern.full((m, k)).tile((r, s)).tap.transformation_dims
    assert a == [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    legacy = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    assert a == list(legacy.transformation_dims)
    # All four operand layouts kernels/linalg.py derives, including the
    # transposed-B and column-major-C variants.
    assert TensorAccessPattern.full((n, k)).tile((t, s)).tap.transformation_dims == [
        (n // t, t * k),
        (k // s, s),
        (t, k),
        (s, 1),
    ]
    assert TensorAccessPattern.full((n, m)).tile(
        (t, r)
    ).inverse().transformation_dims == [
        (n // t, t * m),
        (t, r),
        (m // r, r * t),
        (r, 1),
    ]
    # C operand: the "un-blocking" order that linalg.py hand-writes today.
    c = TensorAccessPattern.full((m, n)).tile((r, t)).inverse()
    assert c.transformation_dims == [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    # Reading a tile-blocked buffer with that view yields logical row-major order.
    logical = np.arange(m * n).reshape(m, n)
    blocked = logical.reshape(m // r, r, n // t, t).transpose(0, 2, 1, 3).ravel()
    assert (blocked[visited(c)] == logical.ravel()).all()
    # And it is the inverse of tiling: tile the row-major tensor, concatenate
    # the tiles, read back with inverse().
    tiles = TensorAccessPattern.full((m, n)).tile((r, t))
    cat = np.concatenate([visited(tiles[i]) for i in range(len(tiles))])
    assert (cat[visited(c)] == np.arange(m * n)).all()
    # Step order and the walk inside each tile are part of the blocked buffer.
    for grid in (
        TensorAccessPattern.full((8, 12)).tile((2, 3)).order("col"),
        TensorAccessPattern.full((8, 12)).tile((2, 3)).permute_tile((1, 0)),
        TensorAccessPattern.full((8, 12))
        .tile((2, 3))
        .order("col")
        .permute_tile((1, 0)),
        TensorAccessPattern.full((4, 6, 8))
        .tile((2, 3, 4))
        .order((2, 0, 1))
        .permute_tile((1, 2, 0)),
    ):
        cat = np.concatenate([visited(grid[i]) for i in range(len(grid))])
        size = int(np.prod(grid.tap.tensor_dims))
        assert (cat[visited(grid.inverse())] == np.arange(size)).all()


# CHECK-LABEL: partition_and_partial
@construct_test
def partition_and_partial():
    # The per-column chunk idiom: [1, 1, 1, N // k] at offset i * N // k.
    N, k = 4096, 8
    parts = TensorAccessPattern.full((1, N)).partition(k)
    assert len(parts) == k
    for i in range(k):
        assert parts[i] == TensorAccessPattern(
            (1, N), i * (N // k), [1, 1, 1, N // k], [0, 0, 0, 1]
        )
    # Partition along a leading axis keeps the rows contiguous.
    rows = TensorAccessPattern.full((64, 32)).partition(4, dim=0)
    assert rows[1] == TensorAccessPattern((64, 32), 512, [1, 1, 16, 32], [0, 0, 32, 1])
    # A ragged group: 14 tiles, repeat 7 spaced 3 apart -> repeat capped at 5,
    # then 5, 5, 4 tiles for the three groups.
    g = (
        TensorAccessPattern.full((3, 28))
        .tile((3, 2))
        .group((1, 7), steps=(1, 3), partial=True)
    )
    assert g.grid_shape == [1, 3]
    assert [g[i].sizes[0] for i in range(3)] == [5, 5, 4]
    base = np.arange(3 * 28).reshape(3, 28)
    for j in range(3):
        tiles = list(range(j, 14, 3))[: g[j].sizes[0]]
        want = np.concatenate([base[:, t * 2 : (t + 1) * 2].ravel() for t in tiles])
        assert (visited(g[j]) == want).all()
    try:
        TensorAccessPattern.full((3, 28)).tile((3, 2)).group((1, 7), steps=(1, 3))
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
        TensorAccessPattern.full((8, 8)).split(0, 3)
        assert False
    except ValueError as e:
        assert "not divisible" in str(e)
