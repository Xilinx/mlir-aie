#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The decode's shared 8-lane int8 GEMV array, which runs every projection.

Eight cores on columns 0, 1, 6 and 7 (two per column) each own one shim MM2S
weight lane and an eighth of every projection's output rows. Per layer the
cores run q, k, v, o, gate, up and down back to back on a broadcast
activation; each lane's weights for a layer are one contiguous DDR region in
consumption order. The cores run aie::mmul (kernels/llama_gemv_mma.cc): a
slot (`MMA_SLOT`) is eight rows by 1024 columns, then their int32 biases and
fp32 scales; each core spreads an activation into its mma tile
(llama_mma_spread) and runs a projection's row groups (`mma_rows`,
`mma_down`). aie2_decode.py builds the decode from these.
"""

import llama_kernels
import numpy as np
from aie.dialects.arith import (  # pyright: ignore[reportMissingImports]
    CmpIPredicate,
    andi,
    cmpi,
    constant,
    extui,
    index_cast,
    ori,
    shli,
    shrui,
)
from aie.extras.types import i32  # pyright: ignore[reportMissingImports]
from aie.iron.controlflow import range_

D = 2048
HD = 8192
N_CORES = 8
CORE_TILES = [(c, r) for c in (0, 1, 6, 7) for r in (2, 3)]
OUT_BYTES = 4096
# The scratchpad keeps 30 bits a word, so the sampler's 32-bit temperature
# bits and seed take one word each for their low 30 bits, and a third word
# holds both their top two bits and top_k.
SAMPLE_PARAMS = ("samp0", "samp1", "samp2")


def sample_param_values(temperature, top_k, seed, kset):
    """The sampler's ScratchpadParameter values for one dispatch.

    Args:
        temperature: float32; at or below 0 the sampler is greedy.
        top_k: 0 (the whole set) or 1..kset.
        seed: The 32-bit seed.
        kset: The resident set's size.

    Returns:
        `SAMPLE_PARAMS`' values by name, as `read_sample_params` reads them.

    Raises:
        ValueError: top_k is not 0 (the whole set) or 1..kset.
    """
    if not 0 <= top_k <= kset:
        raise ValueError(f"top_k must be 0 (the whole set) or 1..{kset}, got {top_k}")
    t = int(np.float32(temperature).view(np.uint32))
    s = int(seed) & 0xFFFFFFFF
    low = (1 << 30) - 1
    return dict(
        samp0=t & low,
        samp1=s & low,
        samp2=(t >> 30) | (s >> 30) << 2 | int(top_k) << 4,
    )


def read_sample_params(params, words):
    """In a core body: the sampler's [temperature bits, top_k, seed] into
    params[0..2] from the `SAMPLE_PARAMS` words."""
    w0, w1, w2 = (p.read() for p in words)
    c3, c30 = constant(i32(), 3), constant(i32(), 30)
    params[0] = ori(w0, shli(andi(w2, c3), c30))
    params[1] = shrui(w2, constant(i32(), 4))
    params[2] = ori(w1, shli(andi(shrui(w2, constant(i32(), 2)), c3), c30))


PROJECTIONS = {
    "q": (D, D, "f32"),
    "k": (D, 512, "f32"),
    "v": (D, 512, "f32"),
    "o": (D, D, "f32"),
    "g": (D, HD, "i8"),
    "u": (D, HD, "f32"),
    "d": (HD, D, "f32"),
}
LAYER = "qkvogud"
COMB_BYTES = 1056
RESID_BLOCK = 256 * 4 + 64
DOWN_CHUNKS = HD // D
DOWN_ROWS_PER_CORE = D // N_CORES


# kernels/llama_gemv_mma.cc's layouts: a slot is eight weight rows by half
# of K=2048 columns, then their biases and scales; an activation object is
# eight activations' 2048 columns in 8x8 tiles, then their act_scales.
MMA_SLOT = 8 * 1024 + 64
MMA_ACT_BYTES = 8 * D + 64


def lane_slots(order):
    """MMA_SLOTs one core's lane holds for a layer: per projection, each row
    group of eight takes one slot per 1024 columns."""
    return sum(PROJECTIONS[p][1] // N_CORES * PROJECTIONS[p][0] // 8192 for p in order)


def mma_rows(
    proj,
    tile,
    c_w,
    part,
    out,
    kernels,
    gate_inv,
    row0=0,
    m_count=1,
    ostride=0,
    groups=None,
):
    """In a core body: the core's rows of a 2048-column projection for the
    activations in the tile's first m_count slots: activation m's row n at
    out[m * ostride + row0 + n] (fp32, or int8 for gate).

    Args:
        proj: The projection's letter.
        tile: The core's mma activation tile.
        c_w: Its weight lane's consumer.
        part: Its int32 partial sums.
        out: Where the rows go.
        kernels: The bound (llama_mma_acc, llama_mma_f32, llama_mma_i8).
        gate_inv: The gate's 1 / out_scale.
        row0: The first row's index in out.
        m_count: Activations (an MLIR i32 value or an int).
        ostride: Elements between two activations' rows in out.
        groups: Row groups of eight to run (default: all the core's).
    """
    k_acc, k_f, k_i = kernels
    for g in range_(groups or PROJECTIONS[proj][1] // N_CORES // 8):
        w = c_w.acquire(1)
        k_acc(tile, w, part, 0, 0, 1)
        c_w.release(1)
        w = c_w.acquire(1)
        at = _i32(g) * 8 + row0
        if PROJECTIONS[proj][2] == "i8":
            k_i(tile, w, part, out, 0, 1, ostride, at, m_count, gate_inv)
        else:
            k_f(tile, w, part, out, 0, 1, ostride, at, m_count)
        c_w.release(1)


def mma_down(tile, c_w, part, out, kernels, spread_chunk, m_count=1):
    """In a core body: the core's down rows (fp32, activation m's at
    out[256 m + n]) through the tile in four 2048-column chunks; chunks 0-2
    sum into the row groups' partials.

    Args:
        tile: The core's mma activation tile.
        c_w: Its weight lane's consumer.
        part: Its int32 partial sums, eight per row.
        out: Where the rows go.
        kernels: The bound (llama_mma_acc, llama_mma_f32, llama_mma_i8).
        spread_chunk: Called with a chunk's index (an MLIR i32 value or an
            int) to spread that chunk's activations into the tile.
        m_count: Activations (an MLIR i32 value or an int).
    """
    k_acc, k_f, _ = kernels
    groups = PROJECTIONS["d"][1] // N_CORES // 8
    for chunk in range_(DOWN_CHUNKS - 1):
        ci = _i32(chunk)
        spread_chunk(ci)
        first = extui(i32(), cmpi(CmpIPredicate.eq, ci, constant(i32(), 0)))
        for g in range_(groups):
            w = c_w.acquire(1)
            k_acc(tile, w, part, _i32(g), 0, first)
            c_w.release(1)
            w = c_w.acquire(1)
            k_acc(tile, w, part, _i32(g), 1, 0)
            c_w.release(1)
    spread_chunk(DOWN_CHUNKS - 1)
    for g in range_(groups):
        w = c_w.acquire(1)
        k_acc(tile, w, part, _i32(g), 0, 0)
        c_w.release(1)
        w = c_w.acquire(1)
        k_f(tile, w, part, out, _i32(g), 1, DOWN_ROWS_PER_CORE, _i32(g) * 8, m_count)
        c_w.release(1)


def mma_kernels(t_act, t_slot, t_out, t_scratch):
    """The mma kernel object's bound entry points for these object types.

    Args:
        t_act: The activation objects' type, spread from.
        t_slot: The weight slots' type.
        t_out: The output objects' type.
        t_scratch: A second fp32 output buffer's type.

    Returns:
        (llama_mma_spread, (llama_mma_acc, llama_mma_f32, llama_mma_i8) for
        t_out, the same with llama_mma_f32_to for t_scratch, the tile's type,
        the partials' type).
    """
    km = llama_kernels.gemv_mma()
    t_tile = _i8(MMA_ACT_BYTES)
    t_part = _i32a(8 * DOWN_ROWS_PER_CORE)
    k_acc = km.bind("llama_mma_acc", [t_tile, t_slot, t_part] + [np.int32] * 3)
    k_i = km.bind(
        "llama_mma_i8", [t_tile, t_slot, t_part, t_out] + [np.int32] * 5 + [np.float32]
    )
    return (
        km.bind("llama_mma_spread", [t_act, t_tile] + [np.int32] * 5),
        (
            k_acc,
            km.bind("llama_mma_f32", [t_tile, t_slot, t_part, t_out] + [np.int32] * 5),
            k_i,
        ),
        (
            k_acc,
            km.bind(
                "llama_mma_f32_to", [t_tile, t_slot, t_part, t_scratch] + [np.int32] * 5
            ),
            k_i,
        ),
        t_tile,
        t_part,
    )


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def _i32a(n):
    return np.ndarray[(int(n),), np.dtype[np.int32]]


def _i32(idx):
    return index_cast(i32(), idx)
