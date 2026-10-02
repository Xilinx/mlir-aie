# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from Inputs.legacy_tensortiler2d import TensorTiler2D
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import require
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
    """Return the same walk through TensorAccessPattern.gather, for cross-checking."""
    return tap.gather(np.arange(int(np.prod(tap.tensor_dims))).reshape(tap.tensor_dims))


# CHECK-LABEL: full_and_permute
@construct_test
def full_and_permute():
    base = np.arange(6 * 4).reshape(6, 4)
    lay = TensorAccessPattern.full((6, 4))
    assert lay.sizes == (6, 4) and lay.strides == (4, 1) and lay.offset == 0
    assert (visited(lay) == base.ravel()).all()
    assert (tap_order(lay) == base.ravel()).all()

    t = lay.permute((1, 0))
    assert t.sizes == (4, 6) and t.strides == (1, 4)
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
    assert s.sizes == (4, 2, 6) and s.strides == (12, 6, 1)
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
    # A unit dim merges into its neighbour whatever its stride.
    strided = TensorAccessPattern((8,), 0, [4, 1], [2, 5])
    assert strided.merge(0) == TensorAccessPattern((8,), 0, [4], [2])
    assert (visited(strided.merge(0)) == visited(strided)).all()
    # Unit dims disappear; a lone unit dim survives.
    u = TensorAccessPattern((4,), 0, [1, 4, 1], [0, 1, 0])
    assert u.coalesce().sizes == (4,)
    assert TensorAccessPattern((4,), 2, [1], [0]).coalesce().sizes == (1,)
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
    assert grid.sizes == (m // r, n // t, r, t)
    for i in range(m // r):
        for j in range(n // t):
            want = base[i * r : (i + 1) * r, j * t : (j + 1) * t].ravel()
            assert (visited(grid[i, j]) == want).all()
            assert grid[i][j] == grid[i, j]
            assert grid.permute((1, 0, 2, 3))[j, i] == grid[i, j]
    # A tiling's walk is its tiles one after another.
    cat = np.concatenate([visited(tile) for row in grid for tile in row])
    assert (cat == visited(grid)).all()
    # Inside-tile transpose.
    ct = grid.permute((0, 1, 3, 2))
    assert (visited(ct[1, 2]) == base[2:4, 6:9].T.ravel()).all()
    # Rank 3 (a NHWC-style tensor) tiles the same way.
    vol = np.arange(4 * 6 * 8).reshape(4, 6, 8)
    g3 = TensorAccessPattern.full((4, 6, 8)).tile((2, 3, 4))
    assert (visited(g3[1, 0, 1]) == vol[2:4, 0:3, 4:8].ravel()).all()


# CHECK-LABEL: tile_slices
@construct_test
def tile_slices():
    # 16 tiles per row; tiles first, first + 4 of every block of 8.
    m, n, r, t = 4, 32, 2, 2
    base = np.arange(m * n).reshape(m, n)
    grid = TensorAccessPattern.full((m, n)).tile((r, t))
    for row in range(m // r):
        for b in range(2):
            for j in range(4):
                first = b * 8 + j
                want = np.concatenate(
                    [
                        base[row * r : (row + 1) * r, tile * t : (tile + 1) * t].ravel()
                        for tile in (first, first + 4)
                    ]
                )
                assert (visited(grid[row, first : first + 8 : 4]) == want).all()
    # A block of tiles, walked tile by tile; permute the grid dims to walk it
    # column by column.
    g2 = TensorAccessPattern.full((8, 8)).tile((2, 2))[0:2, 0:2]
    assert g2.sizes == (2, 2, 2, 2) and g2.strides == (16, 2, 8, 1)
    assert g2.permute((1, 0, 2, 3)).strides == (2, 16, 8, 1)
    # Blocks of tiles: split both grid dims, then bring the block dims out.
    tiles = TensorAccessPattern.full((8, 16)).tile((2, 2))
    blocks = tiles.split(0, 2).split(2, 2).permute((0, 2, 1, 3, 4, 5))
    assert blocks.sizes == (2, 4, 2, 2, 2, 2)
    assert blocks[1, 3] == tiles[2:4, 6:8]
    assert blocks.permute((1, 0, 2, 3, 4, 5))[0, 1] == blocks[1, 0]


# CHECK-LABEL: repeat
@construct_test
def repeat():
    lay = TensorAccessPattern.full((1, 16)).repeat(3)
    assert lay.sizes == (3, 1, 16) and lay.strides == (0, 0, 1)
    assert (visited(lay) == np.tile(np.arange(16), 3)).all()
    g = TensorAccessPattern.full((8, 8)).tile((2, 8))[1, 0].repeat(3)
    assert g.sizes == (3, 2, 8) and g.strides == (0, 8, 1) and g.offset == 16


# CHECK-LABEL: matmul_transformation_dims
@construct_test
def matmul_transformation_dims():
    m, k, n, r, s, t = 64, 64, 64, 4, 8, 4
    # A operand: what TensorTiler2D.group_tiler((m,k),(r,s),(m//r,k//s))[0] gave.
    a = TensorAccessPattern.full((m, k)).tile((r, s)).transformation_dims
    assert a == ((m // r, r * k), (k // s, s), (r, k), (s, 1))
    legacy = TensorTiler2D.group_tiler((m, k), (r, s), (m // r, k // s))[0]
    assert a == legacy.transformation_dims
    # All four operand layouts kernels/linalg.py derives, including the
    # transposed-B and column-major-C variants.
    assert TensorAccessPattern.full((n, k)).tile((t, s)).transformation_dims == (
        (n // t, t * k),
        (k // s, s),
        (t, k),
        (s, 1),
    )
    assert TensorAccessPattern.full((n, m)).tile(
        (t, r)
    ).inverse().transformation_dims == (
        (n // t, t * m),
        (t, r),
        (m // r, r * t),
        (r, 1),
    )
    # C operand: the "un-blocking" order that linalg.py hand-writes today.
    c = TensorAccessPattern.full((m, n)).tile((r, t)).inverse()
    assert c.transformation_dims == ((m // r, r * n), (r, t), (n // t, r * t), (t, 1))
    # Reading a tile-blocked buffer with that view yields logical row-major order.
    logical = np.arange(m * n).reshape(m, n)
    blocked = logical.reshape(m // r, r, n // t, t).transpose(0, 2, 1, 3).ravel()
    assert (blocked[visited(c)] == logical.ravel()).all()
    # And it is the inverse of tiling: tile the row-major tensor, concatenate
    # the tiles, read back with inverse().
    tiles = TensorAccessPattern.full((m, n)).tile((r, t))
    cat = np.concatenate([visited(tile) for row in tiles for tile in row])
    assert (cat[visited(c)] == np.arange(m * n)).all()
    # Tile order and the walk inside each tile are part of the blocked buffer.
    for grid in (
        TensorAccessPattern.full((8, 12)).tile((2, 3)).permute((1, 0, 2, 3)),
        TensorAccessPattern.full((8, 12)).tile((2, 3)).permute((0, 1, 3, 2)),
        TensorAccessPattern.full((8, 12)).tile((2, 3)).permute((1, 0, 3, 2)),
        TensorAccessPattern.full((4, 6, 8)).tile((2, 3, 4)).permute((2, 0, 1, 4, 5, 3)),
    ):
        size = int(np.prod(grid.tensor_dims))
        assert (visited(grid)[visited(grid.inverse())] == np.arange(size)).all()


# CHECK-LABEL: partition_and_ragged_slices
@construct_test
def partition_and_ragged_slices():
    # The per-column chunk idiom: [1, 1, 1, N // k] at offset i * N // k.
    N, k = 4096, 8
    parts = TensorAccessPattern.full((N,)).partition(k)
    assert parts.sizes[0] == k
    for i in range(k):
        assert parts[i] == TensorAccessPattern(
            (N,), i * (N // k), [1, 1, 1, N // k], [0, 0, 0, 1]
        )
    # Like np.array_split, partition cuts the leading axis by default.
    rows = TensorAccessPattern.full((64, 32)).partition(4)
    assert rows[1] == TensorAccessPattern((64, 32), 512, [1, 1, 16, 32], [0, 0, 32, 1])
    base = np.arange(64 * 32).reshape(64, 32)
    for axis in (0, -1):
        chunks = TensorAccessPattern.full((64, 32)).partition(4, dim=axis)
        for got, want in zip(chunks, np.array_split(base, 4, axis=axis)):
            assert (visited(got) == want.ravel()).all()
    # Strided slices of 14 tiles, 3 apart: 5, 5, 4 tiles, as in NumPy.
    tiles = TensorAccessPattern.full((3, 28)).tile((3, 2))
    g = [tiles[0, j::3] for j in range(3)]
    assert [x.sizes[0] for x in g] == [5, 5, 4]
    base = np.arange(3 * 28).reshape(3, 28)
    for j in range(3):
        want = np.concatenate(
            [base[:, t * 2 : (t + 1) * 2].ravel() for t in range(j, 14, 3)]
        )
        assert (visited(g[j]) == want).all()


# CHECK-LABEL: ints_stay_ints
@construct_test
def ints_stay_ints():
    t = TensorAccessPattern.full((np.int64(4), 6))[1:, np.int32(1) :: 2]
    assert t.numel == 9 and isinstance(t.numel, int)
    assert all(type(x) is int for x in (t.offset, *t.sizes, *t.strides))
    try:
        require(False, "a guard on a Python bool raises")
        assert False
    except ValueError as e:
        assert "a guard on a Python bool raises" in str(e)
    try:
        TensorAccessPattern.full((8, 8)).split(0, 3)
        assert False
    except ValueError as e:
        assert "not divisible" in str(e)
