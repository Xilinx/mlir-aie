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
    addi,
    andi,
    cmpi,
    constant,
    extui,
    index_cast,
    muli,
    ori,
    shli,
    shrui,
)
from aie.extras.types import i32  # pyright: ignore[reportMissingImports]
from aie.ir import Value  # pyright: ignore[reportMissingImports]
from aie.iron.controlflow import range_
from llama_config import HD, PROJECTIONS, D

N_CORES = 8
CORE_TILES = [(c, r) for c in (0, 1, 6, 7) for r in (2, 3)]
OUT_BYTES = 4096
LAYER = "qkvogud"
# A row's fused qkv outputs in fp32: q 256, k 64, v 64.
QKV_COLS = 384
# A KV group's combined qkv chunk: q 288 B, k and v fp32, cs 256 B.
COMB_BYTES = 1056
# A core's 256 residual rows in fp32, then 64 B of padding.
RESID_BLOCK = 256 * 4 + 64
DOWN_CHUNKS = HD // D
DOWN_ROWS_PER_CORE = D // N_CORES
# kernels/llama_gemv_mma.cc's layouts: a slot is eight weight rows by half
# of K=2048 columns, then their biases and scales; an activation object is
# eight activations' 2048 columns in 8x8 tiles, then their act_scales.
MMA_SLOT = 8 * 1024 + 64
MMA_ACT_BYTES = 8 * D + 64
# The scratchpad keeps 30 bits a word, so the sampler's 32-bit temperature
# bits take one word for their low 30 bits, and a second holds their top two
# bits and top_k.
SAMPLE_PARAMS = ("samp0", "samp1")


def sample_param_values(temperature, top_k, kset):
    """The sampler's ScratchpadParameter values for one dispatch.

    Args:
        `temperature`: float32; at or below 0 the sampler is greedy.
        `top_k`: 0 (the whole set) or 1..kset.
        `kset`: The set's size.

    Returns:
        `SAMPLE_PARAMS`' values by name, as `read_sample_params` reads them.

    Raises:
        ValueError: top_k is not 0 (the whole set) or 1..kset, or the
            temperature is positive but not a normal float32 below 2^126,
            the domain of the sampler's reciprocal.
    """
    if not 0 <= top_k <= kset:
        raise ValueError(f"top_k must be 0 (the whole set) or 1..{kset}, got {top_k}")
    temperature = np.float32(temperature)
    if not (temperature <= 0 or np.finfo(np.float32).tiny <= temperature < 2.0**126):
        raise ValueError(f"temperature {temperature} is not a sampler temperature")
    t = int(temperature.view(np.uint32))
    return dict(samp0=t & ((1 << 30) - 1), samp1=(t >> 30) | int(top_k) << 2)


def read_sample_params(params, words):
    """In a core body: the sampler's [temperature bits, top_k] into params.

    Args:
        `params`: The sampler core's parameter buffer; its words 0 and 1.
        `words`: The `SAMPLE_PARAMS` ScratchpadParameters.
    """
    w0, w1 = (p.read() for p in words)
    params[0] = ori(w0, shli(andi(w1, constant(i32(), 3)), constant(i32(), 30)))
    params[1] = shrui(w1, constant(i32(), 2))


def lane_slots(order):
    """The MMA_SLOTs one core's lane holds for a layer.

    Args:
        `order`: The projections' letters, in the order the layer runs them.

    Returns:
        Per projection, one slot per row group of eight and 1024 columns.
    """
    return sum(PROJECTIONS[p][0] // N_CORES * PROJECTIONS[p][1] // 8192 for p in order)


def mma_rows(
    proj,
    tile,
    c_w,
    part,
    out,
    kernels,
    row0=0,
    m_count: int | Value = 1,
    ostride=0,
    groups=None,
):
    """In a core body: the core's rows of a 2048-column projection.

    For the activations in the tile's first m_count slots: activation m's
    row n at out[m * ostride + row0 + n] (fp32).

    Args:
        `proj`: The projection's letter.
        `tile`: The core's mma activation tile.
        `c_w`: Its weight lane's consumer.
        `part`: Its int32 partial sums.
        `out`: Where the rows go.
        `kernels`: The bound (llama_mma_acc, llama_mma_f32).
        `row0`: The first row's index in out.
        `m_count`: Activations (an MLIR i32 value or an int).
        `ostride`: Elements between two activations' rows in out.
        `groups`: Row groups of eight to run (default: all the core's).
    """
    k_acc, k_f = kernels
    for g in range_(groups or PROJECTIONS[proj][0] // N_CORES // 8):
        w = c_w.acquire(1)
        k_acc(tile, w, part, 0, 0, 1)
        c_w.release(1)
        w = c_w.acquire(1)
        row = addi(
            muli(index_cast(i32(), g), constant(i32(), 8)), constant(i32(), row0)
        )
        k_f(tile, w, part, out, 0, 1, ostride, row, m_count)
        c_w.release(1)


def mma_down(tile, c_w, part, out, kernels, spread_chunk, m_count: int | Value = 1):
    """In a core body: the core's down rows through the tile.

    Activation m's row n goes to out[256 m + n] (fp32), in four 2048-column
    chunks; chunks 0-2 sum into the row groups' partials.

    Args:
        `tile`: The core's mma activation tile.
        `c_w`: Its weight lane's consumer.
        `part`: Its int32 partial sums, eight per row.
        `out`: Where the rows go.
        `kernels`: The bound (llama_mma_acc, llama_mma_f32).
        `spread_chunk`: Called with a chunk's index (an MLIR i32 value or an
            int) to spread that chunk's activations into the tile.
        `m_count`: Activations (an MLIR i32 value or an int).
    """
    k_acc, k_f = kernels
    groups = PROJECTIONS["d"][0] // N_CORES // 8
    for chunk in range_(DOWN_CHUNKS - 1):
        ci = index_cast(i32(), chunk)
        spread_chunk(ci)
        first = extui(i32(), cmpi(CmpIPredicate.eq, ci, constant(i32(), 0)))
        for g in range_(groups):
            w = c_w.acquire(1)
            k_acc(tile, w, part, index_cast(i32(), g), 0, first)
            c_w.release(1)
            w = c_w.acquire(1)
            k_acc(tile, w, part, index_cast(i32(), g), 1, 0)
            c_w.release(1)
    spread_chunk(DOWN_CHUNKS - 1)
    for g in range_(groups):
        w = c_w.acquire(1)
        k_acc(tile, w, part, index_cast(i32(), g), 0, 0)
        c_w.release(1)
        w = c_w.acquire(1)
        k_f(
            tile,
            w,
            part,
            out,
            index_cast(i32(), g),
            1,
            DOWN_ROWS_PER_CORE,
            muli(index_cast(i32(), g), constant(i32(), 8)),
            m_count,
        )
        c_w.release(1)


def mma_kernels(t_act, t_slot, t_out, t_scratch, m_max):
    """The mma kernel object's bound entry points for these object types.

    Args:
        `t_act`: The activation objects' type, spread from.
        `t_slot`: The weight slots' type.
        `t_out`: The output objects' type.
        `t_scratch`: A second fp32 output buffer's type.
        `m_max`: The rows an activation object holds.

    Returns:
        (llama_mma_spread, (llama_mma_acc, llama_mma_f32) for t_out, the same
        with llama_mma_f32_to for t_scratch, the tile's type, the partials'
        type).
    """
    km = llama_kernels.gemv_mma(m_max=m_max, out_bytes=OUT_BYTES)
    t_tile = np.ndarray[(MMA_ACT_BYTES,), np.dtype[np.int8]]
    t_part = np.ndarray[(8 * DOWN_ROWS_PER_CORE,), np.dtype[np.int32]]
    k_acc = km.bind("llama_mma_acc", [t_tile, t_slot, t_part] + [np.int32] * 3)
    return (
        km.bind("llama_mma_spread", [t_act, t_tile] + [np.int32] * 5),
        (
            k_acc,
            km.bind("llama_mma_f32", [t_tile, t_slot, t_part, t_out] + [np.int32] * 5),
        ),
        (
            k_acc,
            km.bind(
                "llama_mma_f32_to", [t_tile, t_slot, t_part, t_scratch] + [np.int32] * 5
            ),
        ),
        t_tile,
        t_part,
    )
