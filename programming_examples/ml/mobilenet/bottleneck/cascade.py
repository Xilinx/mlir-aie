#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Cascade bottleneck blocks (bn13-bn14) for MobileNet V3 IRON API rewrite.

bn13 and bn14 each use 5 compute tiles, 2 of which use cascade streams (put/get
pairs). Same logical shape as a regular 1x1 -> DW -> 1x1+skip block, but each
1x1 conv is split across two cascade-connected compute tiles.

Architecture overview (cascade-split convolutions):
  Layer-1 PUT tile:  1x1-relu conv on first input half, puts onto cascade stream
  Layer-1 GET tile:  reads cascade, runs 1x1-relu on second half → full 960-ch row
  Layer-2 tile:      DW-3x3 depthwise producing two split output fifos
  Layer-3 PUT tile:  1x1-proj on first DW-split half, puts onto cascade stream
  Layer-3 GET tile:  reads cascade, runs 1x1-proj+skip on second split, writes output

Weight delivery:
  L1 and L3 split weights are streamed via ObjectFifos (Shim → MemTile → tiles).
  L2 DW weights are static Buffers (baked into the tile at compile time).

Tile placements live in PLACEMENT["cascade"] in aie2_mobilenet_iron.py.
"""

import numpy as np

from aie.iron import Buffer, ObjectFifo, Worker, kernels
from aie.iron.dataflow.cascadeflow import CascadeFlow
from aie.iron.controlflow import range_

from ._common import load_wts, layer_sf as _layer_sf, skip_sf as _skip_sf
from ..network_spec import block as nsblock

# ---------------------------------------------------------------------------
# Algorithm dimensions — derived from network_spec (bn13 and bn14 share shape)
# ---------------------------------------------------------------------------
_BN13 = nsblock("bn13")
_InW, _InH, _InC = _BN13.layers[0].in_shape  # (7, 7, 80)
_L1_OutC = _BN13.layers[0].out_shape[2]  # 960 expansion channels
_L3_OutC = _BN13.layers[-1].out_shape[2]  # 80 final projection channels

# ---------------------------------------------------------------------------
# Cascade-split parameters (placement strategy, not algorithm shape)
# ---------------------------------------------------------------------------
_InputSplit = 2  # cascade splits: each cascade tile handles half the channels
_OutputSplit = 2
_OutputSplit2 = 2
_L1_SplitC = _L1_OutC // _InputSplit  # 480 channels per cascade tile
_OC8 = _L1_OutC // (8 * _OutputSplit)  # inner loop count for L1 kernel  = 60
_OC8_out = _L3_OutC // (8 * _OutputSplit2)  # inner loop count for L3 kernel = 5

# ---------------------------------------------------------------------------
# Weight sizes
# ---------------------------------------------------------------------------
_l1_split_wts_sz = (_InC // _InputSplit) * (_L1_OutC // _OutputSplit)  # 40*480
_l2_wts_sz = 3 * 3 * _L1_OutC * 1  # 8640
_l3_split_wts_sz = (_L1_OutC // _InputSplit) * (_L3_OutC // _OutputSplit2)  # 19200

# Full L1/L3 weight tensors (host → MemTile, then split into halves).
_l1_full_wts_sz = _InC * _L1_OutC  # 80*960 = 76800
_l3_full_wts_sz = _L1_OutC * _L3_OutC  # 960*80 = 76800


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_ty_act_in = np.ndarray[(_InW, 1, _InC), np.dtype[np.int8]]
_ty_l1_out_split = np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]]
_ty_l1_out_full = np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]]
_ty_l1_split_wts = np.ndarray[(_l1_split_wts_sz,), np.dtype[np.int8]]
_ty_l2_wts = np.ndarray[(_l2_wts_sz,), np.dtype[np.int8]]
_ty_l3_split_wts = np.ndarray[(_l3_split_wts_sz,), np.dtype[np.int8]]
_ty_act_out = np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]]
_ty_l1_full_wts = np.ndarray[(_l1_full_wts_sz,), np.dtype[np.int8]]
_ty_l3_full_wts = np.ndarray[(_l3_full_wts_sz,), np.dtype[np.int8]]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
def _make_static_wts(data_dir, size, filename, name):
    """Create a static Buffer for weights (L2 DW, baked into tile at compile time)."""
    return Buffer(
        np.ndarray[(size,), np.dtype[np.int8]],
        initial_value=load_wts(data_dir, filename, size),
        name=name,
    )


# ---------------------------------------------------------------------------
# Worker function bodies (shared between bn13 and bn14 — different kernels and tiles)
# ---------------------------------------------------------------------------
# L1 PUT: for each input row, loop over OutputSplit weight tiles and OC8 inner
# iterations, putting partial 1x1 results onto the cascade stream.
def _l1_put_fn(
    of_in,
    wts_fifo,
    k,
    InW,
    InC,
    OutC,
    InputSplit,
    OutputSplit,
    OC8,
    sf1,
):
    for _ in range_(_InH):
        row_in = of_in.acquire(1)
        for WeightIndex in range_(OutputSplit):
            row_wts = wts_fifo.acquire(1)
            for oc in range_(OC8):
                k(row_in, row_wts, InW, InC, OutC, InputSplit, WeightIndex, 0, oc)
            wts_fifo.release(1)
        of_in.release(1)


# L1 GET: for each input row, accumulates cascade and produces full L1 output row.
def _l1_get_fn(
    of_in,
    of_out,
    wts_fifo,
    k,
    InW,
    InC,
    OutC,
    InputSplit,
    OutputSplit,
    OC8,
    sf1,
):
    for _ in range_(_InH):
        row_in = of_in.acquire(1)
        row_out = of_out.acquire(1)
        for WeightIndex in range_(OutputSplit):
            row_wts = wts_fifo.acquire(1)
            for oc in range_(OC8):
                k(
                    row_in,
                    row_wts,
                    row_out,
                    InW,
                    InC,
                    OutC,
                    sf1,
                    InputSplit,
                    OutputSplit,
                    WeightIndex,
                    0,
                    oc,
                )
            wts_fifo.release(1)
        of_in.release(1)
        of_out.release(1)


# L2 DW: depthwise 3x3 producing two split-channel output rows.
def _l2_fn(of_in, of_out_first, of_out_second, wts_buf, k, InW, OutC2, sf2):
    def _dw(top, mid, bot, border):
        row_out_a = of_out_first.acquire(1)
        row_out_b = of_out_second.acquire(1)
        k(
            top,
            mid,
            bot,
            wts_buf,
            row_out_a,
            row_out_b,
            InW,
            1,
            OutC2,
            3,
            3,
            border,
            sf2,
            0,
        )
        of_out_first.release(1)
        of_out_second.release(1)

    # ── preamble: top row (zero-pad above) ──
    rows = of_in.acquire(2)
    _dw(rows[0], rows[0], rows[1], border=0)
    # ── middle rows ──
    for _ in range_(_InH - 2):
        rows = of_in.acquire(3)
        _dw(rows[0], rows[1], rows[2], border=1)
        of_in.release(1)
    # ── postamble: last row (zero-pad below) ──
    rows = of_in.acquire(2)
    _dw(rows[0], rows[1], rows[1], border=2)
    of_in.release(2)


# L3 PUT: reads first DW split half, puts partial projection result onto cascade.
def _l3_put_fn(
    of_in,
    wts_fifo,
    k,
    InW,
    OutC2,
    OutC3,
    InputSplit,
    OutputSplit2,
    OC8_out,
    sf3,
):
    for _ in range_(_InH):
        row_in = of_in.acquire(1)
        for WeightIndex in range_(OutputSplit2):
            row_wts = wts_fifo.acquire(1)
            for oc in range_(OC8_out):
                k(
                    row_in,
                    row_wts,
                    InW,
                    OutC2,
                    OutC3,
                    InputSplit,
                    WeightIndex,
                    0,
                    oc,
                )
            wts_fifo.release(1)
        of_in.release(1)


# L3 GET: reads second DW split half + cascade + skip → final output row.
def _l3_get_fn(
    of_in,
    skip_in,
    act_out,
    wts_fifo,
    k,
    InW,
    OutC2,
    OutC3,
    InputSplit,
    OutputSplit2,
    OC8_out,
    sf3,
    sfAdd,
):
    for _ in range_(_InH):
        row_in = of_in.acquire(1)
        row_out = act_out.acquire(1)
        skip_row = skip_in.acquire(1)
        for WeightIndex in range_(OutputSplit2):
            row_wts = wts_fifo.acquire(1)
            for oc in range_(OC8_out):
                k(
                    row_in,
                    row_wts,
                    row_out,
                    skip_row,
                    InW,
                    OutC2,
                    OutC3,
                    sf3,
                    sfAdd,
                    InputSplit,
                    OutputSplit2,
                    WeightIndex,
                    0,
                    oc,
                )
            wts_fifo.release(1)
        of_in.release(1)
        act_out.release(1)
        skip_in.release(1)


# ---------------------------------------------------------------------------
# build_cascade — one full cascade block (5 compute workers + 2 weight fifos)
# ---------------------------------------------------------------------------
def build_cascade(blk, act_in, skip_in, sf, *, data_dir, tiles):
    """One full cascade block (5 compute workers + 2 weight fifos).

    blk:         Block from network_spec.NETWORK (bn13 or bn14).
    act_in:      activation input ObjectFifo (drives L1 PUT and L1 GET)
    skip_in:     ObjectFifo whose .cons() forwards a skip row to L3 GET
                 (often the same as act_in; for bn14 it's bn13's output)
    tiles:       dict with keys: l1_put, l1_get, l2, l3_put, l3_get,
                                 mem_l1, mem_l3, mem_skip
    Returns (out_fifo, wts_l1_full, wts_l3_full, [workers]).
    """
    name = blk.name
    # Module-level _InW/_InH/_InC/_L1_OutC/_L3_OutC are derived from bn13's
    # shape (line 39). bn14 reuses the same cascade topology because it has
    # the same shape — enforce that here instead of leaving it implicit.
    assert blk.layers[0].in_shape == (_InW, _InH, _InC), (
        f"cascade builder is shape-locked to bn13 ({_InW},{_InH},{_InC}); "
        f"{name} has in_shape {blk.layers[0].in_shape}"
    )
    assert blk.layers[0].out_shape[2] == _L1_OutC, (
        f"cascade builder expects L1 out_c={_L1_OutC}; "
        f"{name} layer[0] out_c={blk.layers[0].out_shape[2]}"
    )
    assert blk.layers[-1].out_shape[2] == _L3_OutC, (
        f"cascade builder expects L3 out_c={_L3_OutC}; "
        f"{name} layer[-1] out_c={blk.layers[-1].out_shape[2]}"
    )
    block_index = int(name[2:])  # bn13 / bn14 → 13 / 14
    s1, s2, s3 = (_layer_sf(blk, sf, i) for i in (0, 1, 2))
    s_add = _skip_sf(blk, sf)

    k_l1_put = kernels.bn_conv2dk1_partial_put_i8(
        input_width=_InW,
        input_channels=_InC,
        weight_count=_l1_split_wts_sz,
        block_index=block_index,
    )
    k_l1_get = kernels.bn_conv2dk1_partial_get_relu_i8(
        input_width=_InW,
        input_channels=_InC,
        output_channels=_L1_OutC,
        weight_count=_l1_split_wts_sz,
        block_index=block_index,
    )
    k_l2_dw = kernels.bn_conv2dk3_dw_out_split(
        input_width=_InW,
        input_channels=_L1_OutC,
        output_split_channels=_L1_SplitC,
        block_index=block_index,
    )
    k_l3_put = kernels.bn_conv2dk1_input_split_partial_put_ui8(
        input_width=_InW,
        input_channels=_L1_SplitC,
        weight_count=_l3_split_wts_sz,
        block_index=block_index,
    )
    k_l3_get = kernels.bn_conv2dk1_input_split_partial_skip_get(
        input_width=_InW,
        input_channels=_L1_SplitC,
        output_channels=_L3_OutC,
        weight_count=_l3_split_wts_sz,
        block_index=block_index,
    )

    # Streaming weight fifos (Shim → MemTile → split → put/get tiles)
    wts_l1_full = ObjectFifo(_ty_l1_full_wts, depth=1)
    wts_l1_put_h, wts_l1_get_h = wts_l1_full.cons().split(
        offsets=[0, _l1_full_wts_sz // 2],
        depths=[1, 1],
        obj_types=[_ty_l1_split_wts, _ty_l1_split_wts],
        tile=tiles["mem_l1"],
        repeat_counts=[_InH, _InH],
    )
    wts_l3_full = ObjectFifo(_ty_l3_full_wts, depth=1)
    wts_l3_put_h, wts_l3_get_h = wts_l3_full.cons().split(
        offsets=[0, _l3_full_wts_sz // 2],
        depths=[1, 1],
        obj_types=[_ty_l3_split_wts, _ty_l3_split_wts],
        tile=tiles["mem_l3"],
        repeat_counts=[_InH, _InH],
    )

    # L2 DW weights are static (compile-time bake-in)
    l2_wts = _make_static_wts(
        data_dir, _l2_wts_sz, f"{name}_2_chain.txt", f"{name}_2_wts_static"
    )

    of_l1_l2 = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]],
        depth=4,
        via_DMA=True,
    )
    of_l2_l3_first = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2,
    )
    of_l2_l3_second = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2,
    )
    out_fifo = ObjectFifo(
        np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]],
        depth=2,
    )

    # Pre-establish .cons() ordering to match expected MLIR placement.
    l1_put_cons = act_in.cons()
    l1_get_cons = act_in.cons()
    # depth=6 on the cons handle lets the skip path buffer enough rows to
    # outlive the 5-tile cascade pipeline lag (l1_put → l1_get → l2 → l3_put → l3_get).
    skip_fifo = skip_in.cons(depth=6).forward(depth=2, tile=tiles["mem_skip"])

    bws = [
        Worker(
            _l1_put_fn,
            fn_args=[
                l1_put_cons,
                wts_l1_put_h.cons(),
                k_l1_put,
                _InW,
                _InC,
                _L1_OutC,
                _InputSplit,
                _OutputSplit,
                _OC8,
                s1,
            ],
            tile=tiles["l1_put"],
        ),
        Worker(
            _l1_get_fn,
            fn_args=[
                l1_get_cons,
                of_l1_l2.prod(),
                wts_l1_get_h.cons(),
                k_l1_get,
                _InW,
                _InC,
                _L1_OutC,
                _InputSplit,
                _OutputSplit,
                _OC8,
                s1,
            ],
            tile=tiles["l1_get"],
        ),
        Worker(
            _l2_fn,
            fn_args=[
                of_l1_l2.cons(),
                of_l2_l3_first.prod(),
                of_l2_l3_second.prod(),
                l2_wts,
                k_l2_dw,
                _InW,
                _L1_OutC,
                s2,
            ],
            tile=tiles["l2"],
        ),
        Worker(
            _l3_put_fn,
            fn_args=[
                of_l2_l3_first.cons(),
                wts_l3_put_h.cons(),
                k_l3_put,
                _InW,
                _L1_OutC,
                _L3_OutC,
                _InputSplit,
                _OutputSplit2,
                _OC8_out,
                s3,
            ],
            tile=tiles["l3_put"],
        ),
        Worker(
            _l3_get_fn,
            fn_args=[
                of_l2_l3_second.cons(),
                skip_fifo.cons(),
                out_fifo.prod(),
                wts_l3_get_h.cons(),
                k_l3_get,
                _InW,
                _L1_OutC,
                _L3_OutC,
                _InputSplit,
                _OutputSplit2,
                _OC8_out,
                s3,
                s_add,
            ],
            tile=tiles["l3_get"],
        ),
    ]
    # Cascade flows: L1 put→get and L3 put→get share streams between adjacent tiles.
    CascadeFlow(bws[0], bws[1])
    CascadeFlow(bws[3], bws[4])
    return out_fifo, wts_l1_full, wts_l3_full, bws


# ---------------------------------------------------------------------------
# Public entry point — bn13 + bn14, dispatched from network_spec.NETWORK
# ---------------------------------------------------------------------------
def cascade_bottlenecks(
    act_in: ObjectFifo,
    sf: dict,
    *,
    placement: dict,
    data_dir: str,
) -> tuple:
    """Build bn13 and bn14 from network_spec.NETWORK + scale-factor JSON.

    Each block has 5 compute workers and 2 cascade stream put/get pairs.
    Cascade flows are constructed inside `build_cascade` and self-register on
    their source workers; they're picked up automatically by Program.resolve().

    Returns:
        (workers, act_bn14_out, wts_fifos) where wts_fifos is the 4 full-weight
        ObjectFifos the host DMA writes into (bn13_l1, bn13_l3, bn14_l1, bn14_l3).
    """
    workers = []

    # bn13: cascade-split bottleneck (5 compute workers).
    act_bn13_out, bn13_wts_l1_full, bn13_wts_l3_full, bn13_workers = build_cascade(
        nsblock("bn13"),
        act_in=act_in,
        skip_in=act_in,
        sf=sf,
        data_dir=data_dir,
        tiles=placement["bn13"],
    )
    workers += bn13_workers

    # bn14: cascade-split bottleneck (5 compute workers, skip = bn13 output).
    act_bn14_out, bn14_wts_l1_full, bn14_wts_l3_full, bn14_workers = build_cascade(
        nsblock("bn14"),
        act_in=act_bn13_out,
        skip_in=act_bn13_out,
        sf=sf,
        data_dir=data_dir,
        tiles=placement["bn14"],
    )
    workers += bn14_workers

    wts_fifos = [
        bn13_wts_l1_full,
        bn13_wts_l3_full,
        bn14_wts_l1_full,
        bn14_wts_l3_full,
    ]
    return workers, act_bn14_out, wts_fifos
