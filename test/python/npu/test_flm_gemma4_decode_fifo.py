# test_flm_gemma4_decode_fifo.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings
"""Projection arithmetic and shared-memory row handoff, without kernel locks."""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Port,
    Program,
    Runtime,
    Transport,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from ml_dtypes import bfloat16

pytestmark = pytest.mark.supported_devices("npu2")
BF16 = np.dtype[bfloat16]
GEOMETRY = kernels.FLM_GEMMA4_E2B_DECODE
D = GEOMETRY.model_dim
RELATIONS = {
    "south": ((1, 3), (1, 2)),
    "north": ((1, 2), (1, 3)),
    "west": ((1, 2), (0, 2)),
}


def _tiles(relation):
    (cc, cr), (nc, nr) = RELATIONS[relation]
    core = Tile(cc, cr, tile_type=AIETileType.CoreTile)
    near = (
        core if (nc, nr) == (cc, cr) else Tile(nc, nr, tile_type=AIETileType.CoreTile)
    )
    return Tile(cc, 0, tile_type=AIETileType.ShimNOCTile), core, near


M_BLOCK, K_BLOCK = 32, 256
BLOCK_BYTES = M_BLOCK * K_BLOCK * 5 // 8
Y_WORDS = 2 * M_BLOCK + 16
MVM_CORES = 16


def _proj_main_shapes():
    # IS_SWA = 1 and SKIP_KV = 0 on the double-wide-MLP E2B build: qkv, o,
    # up/gate and down, each this core's 1/16 share of the rows.
    swa_q = GEOMETRY.num_attn_heads * GEOMETRY.swa_dh
    swa_kv = GEOMETRY.num_kv_heads * GEOMETRY.swa_dh
    inter = GEOMETRY.intermediate_size
    assert GEOMETRY.double_wide_mlp
    return [
        ((swa_q + 2 * swa_kv) // MVM_CORES, D),
        (D // MVM_CORES, swa_q),
        (2 * inter // MVM_CORES, D),
        (D // MVM_CORES, inter),
    ]


SHAPES = _proj_main_shapes()
PROJ_BLOCKS = sum((m // M_BLOCK) * (k // K_BLOCK) for m, k in SHAPES)
Y_SLOTS = sum(m // M_BLOCK for m, _ in SHAPES)
X_WORDS = sum((m // M_BLOCK) * k for m, k in SHAPES)


@iron.jit
def proj_main_design(X: In, W: In, Y: Out, *, relation: CompileTime[str]):
    shim, core, near = _tiles(relation)
    k = kernels.flm_gemma4_decode_proj_main(geometry=GEOMETRY)
    w_ty, x_ty, acc_ty, sums_ty = k.arg_types()[:4]
    row_ty = k.proj_store.arg_types()[0]
    y_ty = k.proj_pack.arg_types()[0]
    x_fifo, w_fifo = ObjectFifo(x_ty), ObjectFifo(w_ty)
    row_fifo = ObjectFifo(row_ty, delegate_tile=near, transport=Transport.shared_mem())
    out_fifo = ObjectFifo(y_ty)
    acc = Buffer(type=acc_ty, tile=core)
    sums = Buffer(type=sums_ty, tile=core)
    empty = Buffer(type=acc_ty, tile=near)

    def project(x_h, w_h, row_h, acc, sums, mac, zero, store):
        for m, k in SHAPES:
            for i in range_(m // M_BLOCK):
                zero(acc)
                for j in range_(k // K_BLOCK):
                    w, x = w_h.acquire(1), x_h.acquire(1)
                    mac(w, x, acc, sums, i, j)
                    w_h.release(1)
                    x_h.release(1)
                store(row_h.acquire(1), acc)
                row_h.release(1)

    def assemble(row_h, out_h, empty, zero, pack):
        zero(empty)
        for _ in range_(Y_SLOTS):
            pack(out_h.acquire(1), empty, row_h.acquire(1), 4)
            row_h.release(1)
            out_h.release(1)

    workers = [
        Worker(
            project,
            [
                x_fifo.cons(),
                w_fifo.cons(),
                row_fifo.prod(),
                acc,
                sums,
                k,
                k.proj_zero,
                k.proj_store,
            ],
            tile=core,
        ),
        Worker(
            assemble,
            [row_fifo.cons(), out_fifo.prod(), empty, k.proj_zero, k.proj_pack],
            tile=near,
        ),
    ]

    def sequence(x, w, y, x_h, w_h, y_h):
        x_h.fill(x)
        w_h.fill(w)
        y_h.drain(y, wait=True)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(X_WORDS,), BF16],
            np.ndarray[(PROJ_BLOCKS * BLOCK_BYTES // 2,), BF16],
            np.ndarray[(Y_SLOTS * Y_WORDS,), BF16],
            x_fifo.prod(tile=shim, port=Port.dma(0)),
            w_fifo.prod(tile=shim, port=Port.dma(1)),
            out_fifo.cons(tile=shim, port=Port.dma(0)),
        ],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _q4nx_block(codes, scales, mins):
    """Pack one 32 by 256 q4nx block: scales and minima [group, row], then codes.

    The codes sit in strips of 16 rows, each strip column-major, two to a
    byte, low nibble first.
    """
    order = codes.reshape(2, 16, 8, 4, 8).transpose(0, 2, 3, 4, 1).reshape(-1)
    packed = (order[0::2] | (order[1::2] << 4)).astype(np.uint8)
    head = np.concatenate([scales.reshape(-1), mins.reshape(-1)]).astype(bfloat16)
    return np.concatenate([head.view(np.uint8), packed])


def _proj_main_inputs(rng):
    # Codes in [0, 3], x in {-1, 0, 1}, power-of-two scales and integer
    # minima: each group's code sum (at most 96) and x sum (at most 32) are
    # exact in bf16, and every float32 partial sum is exact.
    x_stream, w_stream, expected = [], [], []
    for m, k in SHAPES:
        x = rng.choice([-1.0, 0.0, 1.0], k).astype(bfloat16)
        y = np.zeros(m)
        for i in range(m // M_BLOCK):
            for j in range(k // K_BLOCK):
                codes = rng.integers(0, 4, (M_BLOCK, K_BLOCK))
                scales = rng.choice([0.5, 1.0, 2.0], (K_BLOCK // 32, M_BLOCK))
                mins = rng.integers(-2, 3, (K_BLOCK // 32, M_BLOCK)).astype(float)
                w_stream.append(_q4nx_block(codes, scales, mins))
                xs = x[j * K_BLOCK : (j + 1) * K_BLOCK].astype(np.float64)
                x_stream.append(x[j * K_BLOCK : (j + 1) * K_BLOCK])
                groups = np.arange(K_BLOCK) // 32
                dequant = scales[groups].T * codes + mins[groups].T
                y[i * M_BLOCK : (i + 1) * M_BLOCK] += dequant @ xs
        expected.append(y)
    return (
        np.concatenate(x_stream),
        np.concatenate(w_stream).view(bfloat16),
        np.concatenate(expected),
    )


@pytest.mark.parametrize("relation", list(RELATIONS))
def test_projection_row_fifo(relation):
    rng = np.random.default_rng(37)
    x, w, expected = _proj_main_inputs(rng)
    y = iron.zeros(Y_SLOTS * Y_WORDS, dtype=bfloat16, device="npu")
    proj_main_design(
        iron.tensor(x, dtype=bfloat16, device="npu"),
        iron.tensor(w, dtype=bfloat16, device="npu"),
        y,
        relation=relation,
    )
    y.to("cpu")
    words = y.numpy().reshape(Y_SLOTS, Y_WORDS)
    np.testing.assert_array_equal(words[:, 16 : 16 + M_BLOCK], 0)
    np.testing.assert_array_equal(words[:, 14:16].copy().view(np.uint32), 4)
    # The receiving core assembles its peer's row into the second slot.
    got = y.numpy().reshape(Y_SLOTS, Y_WORDS)[:, 16 + M_BLOCK :].astype(np.float64)
    got = got.reshape(-1)
    _, e = np.frexp(np.abs(expected))
    ulp = np.ldexp(1.0, e - 8)
    bad = np.abs(got - expected) > ulp
    assert not bad.any(), (
        f"{relation}: {bad.sum()} of {bad.size} outputs off by more than one "
        f"bf16 ulp; first at {np.flatnonzero(bad)[0]}: got "
        f"{got[bad][0]}, expected {expected[bad][0]}"
    )
