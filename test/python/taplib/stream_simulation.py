# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from util import construct_test

# RUN: %python %s | FileCheck %s

# A data path is a chain of DMAs: each one walks an object onto a stream
# (gather) or off it into the next object (scatter). Simulating the
# chain with the same patterns the design uses shows what every core receives.


def objects(stream, size):
    """Split a stream into the consecutive objects a fifo of ``size`` elements holds."""
    assert stream.size % size == 0
    return stream.reshape(-1, size)


# CHECK-LABEL: transposes_combined_chain
@construct_test
def transposes_combined_chain():
    """basic/transposes --strategy=combined.

    The memtile block-shuffles each tile so the kernel only transposes s x s
    sub-tiles.
    """
    M, K, m, n, s = 64, 64, 16, 16, 8
    host = np.arange(M * K).reshape(M, K)
    shim = TensorAccessPattern((M, K), 0, [M // m, K // n, m, n], [m * K, n, K, 1])
    to_core = TensorAccessPattern((m, n), 0, [m // s, s, n // s, s], [s, m, s * m, 1])
    tiles = objects(shim.gather(host), m * n)
    assert len(tiles) == (M // m) * (K // n)
    for t, obj in enumerate(tiles):
        i, j = divmod(t, K // n)
        core = to_core.gather(obj)
        # The kernel transposes every s x s block in place; the result must
        # be the transposed tile.
        blocks = core.reshape(m // s, s, n // s, s).transpose(0, 2, 3, 1)
        kernel_out = blocks.transpose(0, 2, 1, 3).reshape(m, n)
        assert (kernel_out == host[i * m : (i + 1) * m, j * n : (j + 1) * n].T).all()
    print("transposes chain: every tile arrives block-transposed")
    # CHECK: transposes chain: every tile arrives block-transposed


# CHECK-LABEL: memtile_4d_core_3d_chain
@construct_test
def memtile_4d_core_3d_chain():
    """test/npu-xrt/dma_complex_dims.

    Linear shim, 4-D memtile MM2S, 3-D core S2MM; test.cpp derives the
    expected order by hand.
    """
    m, k, K, r, s = 32, 32, 64, 4, 8
    host = np.arange(m * K)  # pre-tiled on the host: [tile][m][k]
    memtile = TensorAccessPattern((m, K), 0, [K // k, k // s, m, s], [m * k, s, k, 1])
    core_in = TensorAccessPattern((m, k), 0, [k // s, m // r, r * s], [r * s, r * k, 1])
    stream = memtile.gather(TensorAccessPattern.full((m, K)).gather(host))
    cores = [core_in.scatter(obj) for obj in objects(stream, m * k)]
    assert len(cores) == K // k
    for tile_k, core in enumerate(cores):
        want = [
            host[tile_k * m * k + ii * r * k + jj * s + r_ii * k + s_jj]
            for ii in range(m // r)
            for jj in range(k // s)
            for r_ii in range(r)
            for s_jj in range(s)
        ]
        assert (core.reshape(-1) == np.array(want)).all(), tile_k
    print("dma_complex_dims chain matches test.cpp")
    # CHECK: dma_complex_dims chain matches test.cpp


# CHECK-LABEL: matmul_a_chain
@construct_test
def matmul_a_chain():
    """The whole_array A operand.

    Host row of tiles -> memtile split rows -> (r x s) microtile stream into
    each core.
    """
    M, K, m, k, r, s, rows = 128, 64, 32, 32, 4, 8, 2
    host = np.arange(M * K).reshape(M, K)
    A_tiles = TensorAccessPattern.full((M, K)).tile((m * rows, k))
    micro = TensorAccessPattern.full((m, k)).tile((r, s))
    for tile, obj in enumerate(objects(A_tiles[0].gather(host), m * rows * k)):
        for row in range(rows):
            # The memtile splits its object into one m x k block per core row.
            block = obj[row * m * k : (row + 1) * m * k]
            core = micro.gather(block).reshape(m, k)
            want = host[row * m : (row + 1) * m, tile * k : (tile + 1) * k]
            want = want.reshape(m // r, r, k // s, s).transpose(0, 2, 1, 3)
            assert (core == want.reshape(m, k)).all(), (tile, row)
    print("matmul A chain: cores receive (r x s) microtiles")
    # CHECK: matmul A chain: cores receive (r x s) microtiles


# CHECK-LABEL: round_trip
@construct_test
def round_trip():
    # scatter undoes gather for any walk that visits every element once.
    host = np.arange(6 * 8).reshape(6, 8)
    for tap in (
        TensorAccessPattern.full((6, 8)).T,
        TensorAccessPattern.full((6, 8)).tile((3, 4)),
        TensorAccessPattern.full((6, 8)).tile((2, 2)).inverse(),
    ):
        assert (tap.scatter(tap.gather(host)) == host).all()
    # A walk that visits half the tensor leaves the rest of ``out`` alone.
    half = TensorAccessPattern.full((6, 8))[:, :4]
    out = half.scatter(half.gather(host), out=np.full((6, 8), -1))
    assert (out[:, :4] == host[:, :4]).all() and (out[:, 4:] == -1).all()
    try:
        half.scatter(np.arange(5))
        assert False
    except ValueError:
        pass
    print("round trip ok")
    # CHECK: round trip ok


# CHECK-LABEL: padded_gather
@construct_test
def padded_gather():
    """basic/dma_padding: a memtile MM2S pads the stream it emits."""
    real, before, after = 8, 4, 4
    host = np.arange(real)
    padded = TensorAccessPattern.full((real,)).pad([(before, after)])
    assert padded.padding == ((before, after),)
    assert padded.padded_sizes == (before + real + after,)
    assert padded.numel == real
    assert padded.transformation_dims == ((real, 1),)
    got = padded.gather(host, pad_value=-1)
    assert got.tolist() == [-1] * before + host.tolist() + [-1] * after
    # Two dimensions: every row of a (rows, cols) tile gets its own pads and
    # whole padded rows surround the block, exactly np.pad's geometry.
    rows, cols, N = 4, 8, 16
    host2 = np.arange(rows * N).reshape(rows, N)
    shim = TensorAccessPattern.full((rows, N))[:, :cols]
    padded2 = TensorAccessPattern.full((rows, cols)).pad([(1, 1), (2, 2)])
    core = padded2.gather(shim.gather(host2), pad_value=-1)
    want = np.pad(host2[:, :cols], [(1, 1), (2, 2)], constant_values=-1)
    assert (core.reshape(rows + 2, cols + 4) == want).all()
    # Padding is applied last, and only by an emitting DMA.
    for bad in (lambda: padded2.T, lambda: padded2.scatter(core)):
        try:
            bad()
            assert False
        except ValueError:
            pass
    print("padding: pads arrive as -1 in the padded object")
    # CHECK: padding: pads arrive as -1 in the padded object
