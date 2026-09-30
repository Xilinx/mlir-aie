# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.helpers.taplib import Layout, TensorAccessPattern
from aie.helpers.taplib.pipeline import Hop, Pipeline
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: transposes_combined_chain
@construct_test
def transposes_combined_chain():
    """Compose basic/transposes --strategy=combined.

    The memtile block-shuffles each tile so the kernel only transposes s x s
    sub-tiles.
    """
    M, K, m, n, s = 64, 64, 16, 16, 8
    host = np.arange(M * K).reshape(M, K)
    tap_in_L3L2 = TensorAccessPattern(
        (M, K), 0, [M // m, K // n, m, n], [m * K, n, K, 1]
    )
    tap_in_L2L1 = TensorAccessPattern(
        (M, K), 0, [m // s, s, n // s, s], [s, m, s * m, 1]
    )
    pipe = (
        Pipeline()
        .shim(tap_in_L3L2)
        .memtile_in((m, n), tap_in_L2L1.transformation_dims)
        .memtile_out((m, n))
        .core_in((m, n))
    )
    assert pipe.check() == []
    objs = pipe.compose()
    assert len(objs) == (M // m) * (K // n)
    t = 0
    for i in range(M // m):
        for j in range(K // n):
            tile = host[i * m : (i + 1) * m, j * n : (j + 1) * n]
            core = objs[t]  # host indices, as stored in the core's (m, n) object
            # The kernel transposes every s x s block in place; the result must
            # be the transposed tile.
            blocks = core.reshape(m // s, s, n // s, s).transpose(0, 2, 3, 1)
            kernel_out = blocks.transpose(0, 2, 1, 3).reshape(m, n)
            assert (kernel_out == tile.T).all(), (i, j)
            t += 1
    print("transposes chain: every tile arrives block-transposed")
    # CHECK: transposes chain: every tile arrives block-transposed


# CHECK-LABEL: memtile_4d_core_3d_chain
@construct_test
def memtile_4d_core_3d_chain():
    """Compose test/npu-xrt/dma_complex_dims.

    Linear shim, 4-D memtile MM2S, 3-D core S2MM; test.cpp derives the
    expected order by hand.
    """
    m, k, K, r, s = 32, 32, 64, 4, 8
    host = np.arange(m * K)  # pre-tiled on the host: [tile][m][k]
    pipe = (
        Pipeline()
        .shim(Layout.full((m, K)))
        .memtile_in((m, K))
        .memtile_out((m, K), [(K // k, m * k), (k // s, s), (m, k), (s, 1)])
        .core_in((m, k), [(k // s, r * s), (m // r, r * k), (r * s, 1)])
    )
    assert pipe.check() == []
    objs = pipe.compose()
    assert len(objs) == K // k
    for tile_k, core in enumerate(objs):
        got = core.reshape(-1)
        want = [
            host[tile_k * m * k + ii * r * k + jj * s + r_ii * k + s_jj]
            for ii in range(m // r)
            for jj in range(k // s)
            for r_ii in range(r)
            for s_jj in range(s)
        ]
        assert (got == np.array(want)).all(), tile_k
    print("dma_complex_dims chain matches test.cpp")
    # CHECK: dma_complex_dims chain matches test.cpp


# CHECK-LABEL: matmul_a_chain
@construct_test
def matmul_a_chain():
    """Compose the whole_array A operand.

    Host group tiling -> memtile split rows -> (r x s) microtile stream into
    each core.
    """
    M, K, m, k, r, s, rows = 128, 64, 32, 32, 4, 8, 2
    host = np.arange(M * K).reshape(M, K)
    A_tiles = Layout.full((M, K)).tile((m * rows, k)).group((1, K // k))
    a_dims = Layout.full((m, k)).tile((r, s)).layout.stream_dims()
    pipe = Pipeline().shim(A_tiles[0]).memtile_in((m * rows, k))
    for row in range(rows):
        pipe.memtile_out(
            (m * rows, k), a_dims, offset=row * m * k, length=m * k, name=f"row{row}"
        )
    pipe.core_in((m, k))
    assert pipe.check() == []
    objs = pipe.compose()
    # One tap moves a (m*rows, K) strip as K//k tiles; each tile splits into
    # `rows` core objects: object index = tile * rows + row.
    assert len(objs) == (K // k) * rows
    for tile in range(K // k):
        for row in range(rows):
            core = objs[tile * rows + row]
            block = host[row * m : (row + 1) * m, tile * k : (tile + 1) * k]
            # Core object holds the (r x s) microtiles consecutively.
            want = (
                block.reshape(m // r, r, k // s, s).transpose(0, 2, 1, 3).reshape(m, k)
            )
            assert (core == want).all(), (tile, row)
    print("matmul A chain: cores receive (r x s) microtiles")
    # CHECK: matmul A chain: cores receive (r x s) microtiles


# CHECK-LABEL: legality_and_coverage
@construct_test
def legality_and_coverage():
    # A core tile cannot wrap 256 in one dimension.
    h = Hop("core_in", (256, 4), [(256, 4), (4, 1)])
    assert any("exceeds wrap 255" in msg for msg in h.issues())
    # Five real dimensions never fit a shim BD.
    h = Hop("shim", (2, 2, 2, 2, 2), [(2, 16), (2, 8), (2, 4), (2, 2), (2, 1)])
    assert any("addressing dimensions" in msg for msg in h.issues())
    # A 2-byte transpose is a sub-word innermost stride.
    h = Hop("shim", (8, 8), [(8, 1), (8, 8)], elem_bytes=2)
    assert any("not a whole" in msg and "granule" in msg for msg in h.issues())
    # A walk that visits half the object is not a fill.
    bad = Pipeline().shim(Layout.full((4, 8))).memtile_in((4, 8), [(2, 8), (8, 1)])
    assert any("visits 16 elements" in msg for msg in bad.check())
    try:
        bad.compose()
        assert False
    except ValueError:
        pass
    # A pure repeat is bounded by the 8-bit queue repeat count; one that steps
    # (a stride on the outermost dimension) also by the 6-bit iteration wrap.
    assert Hop.shim(Layout.full((1, 16)).repeat(100)).issues() == []
    assert any(
        "repeat 257" in m for m in Hop.shim(Layout.full((1, 16)).repeat(257)).issues()
    )
    stepped = Hop.shim(Layout.full((100, 16)).tile((1, 16)).layout)
    assert any("iteration wrap 64" in m for m in stepped.issues())
    # A legal 4-byte transpose is fine.
    assert Hop.shim(Layout.full((8, 8)).permute((1, 0))).issues() == []


# CHECK-LABEL: padded_memtile_out
@construct_test
def padded_memtile_out():
    """Compose basic/dma_padding.

    A memtile MM2S pads the stream it emits; the consumer's object is the
    padded size and the pads arrive as -1.
    """
    real, before, after = 8, 4, 4
    host = np.arange(real)
    padded = Layout.full((real,)).pad([(before, after)])
    assert padded.padded_sizes == [before + real + after]
    assert padded.numel == before + real + after
    assert padded.stream_dims() == [(real, 1)]
    assert padded.pad_dims() == [(before, after)]
    pipe = (
        Pipeline()
        .shim(Layout.full((real,)))
        .memtile_in((real,))
        .memtile_out((real,), padded)
        .core_in((before + real + after,))
    )
    assert pipe.check() == []
    (core,) = pipe.compose()
    assert core.tolist() == [-1] * before + host.tolist() + [-1] * after
    # Two dimensions: every row of a (rows, cols) tile gets its own pads and
    # whole padded rows surround the block, exactly np.pad's geometry.
    rows, cols, N = 4, 8, 16
    host2 = np.arange(rows * N).reshape(rows, N)
    padded2 = Layout.full((rows, cols)).pad([(1, 1), (2, 2)])
    want = np.pad(host2[:, :cols], [(1, 1), (2, 2)], constant_values=-1)
    assert (
        padded2.materialize()
        == np.pad(
            np.arange(rows * cols).reshape(rows, cols),
            [(1, 1), (2, 2)],
            constant_values=-1,
        )
    ).all()
    pipe2 = (
        Pipeline()
        .shim(Layout.full((rows, N))[:, :cols])
        .memtile_in((rows, cols))
        .memtile_out((rows, cols), padded2)
        .core_in((rows + 2, cols + 4))
    )
    assert pipe2.check() == [], pipe2.check()
    (core2,) = pipe2.compose()
    assert (core2 == want).all()
    # Legality: padding lives on memtile_out only, one pair per dim, and the
    # innermost counts must be whole 32-bit words.
    assert any(
        "only available on a memtile_out" in m
        for m in Hop("core_in", (8,), [(8, 1)], pad=[(4, 4)]).issues()
    )
    assert any(
        "entries for" in m
        for m in Hop("memtile_out", (8,), [(8, 1)], pad=[(4, 4), (0, 0)]).issues()
    )
    assert any(
        "granule" in m
        for m in Hop("memtile_out", (8,), [(8, 1)], elem_bytes=1, pad=[(2, 0)]).issues()
    )
    assert Hop("memtile_out", (8,), [(8, 1)], elem_bytes=1, pad=[(4, 8)]).issues() == []
    print("padding: pads arrive as -1 in the padded object")
    # CHECK: padding: pads arrive as -1 in the padded object
