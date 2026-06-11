#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""MobileNet V3 — IRON API rewrite.

Replaces the placed-dialect implementation in aie2_mobilenet.py with the
high-level IRON API.  Computation is organized by block family, each in
its own sibling module under `bottleneck/`:

  init.py      — 3x3 stride-2 input conv
  regular.py   — bn0–bn9  (single compute tile per block)
  pipeline.py  — bn10–bn12 (one tile per layer)
  cascade.py   — bn13–bn14 (split-channel cascade stream, 5 tiles per block)
  post_l1.py   — avg pool + expand 1x1 conv
  post_l2.py   — 4-tile FC1+FC2 (split output channels)

Scale factors are compile-time Python int constants loaded from
scale_factors_final.json and passed directly in Worker fn_args — no RTP
buffers or NpuWriteRTPOp calls are needed.

Usage:
    python3 aie2_mobilenet_iron.py > mobilenet_iron.mlir
"""

import json
import os
import sys
import numpy as np

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2

# Allow importing the local bottleneck/ package when running this file directly.
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from bottleneck._common import tile_kw
from bottleneck.init import init_conv
from bottleneck.regular import regular_bottlenecks
from bottleneck.pipeline import pipeline_bottlenecks
from bottleneck.cascade import cascade_bottlenecks
from bottleneck.post_l1 import post_l1
from bottleneck.post_l2 import post_l2
from network_spec import block as nsblock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "data") + "/"
scale_factor_file = "scale_factors_final.json"

with open(data_dir + scale_factor_file) as f:
    sf = json.load(f)


# ---------------------------------------------------------------------------
# Module-scope dims used by the top-level runtime arg types and the runtime
# sequence (block-local dims live inside the builder modules).
# ---------------------------------------------------------------------------
tensorInW, tensorInH, tensorInC = nsblock("init").layers[0].in_shape
post_L1_OutW, post_L1_OutH, _ = nsblock("post_l1").layers[0].out_shape
post_L2_InC = nsblock("post_l2").layers[0].in_shape[2]
post_L2_OutC = nsblock("post_l2").layers[-1].out_shape[2]


# Physical tile placement lives in placement.py (algorithm/mapping split).
# When --no-placement is passed, builders receive no placement and the design
# is emitted unplaced for the SA placer to handle at compile time.
from placement import PLACEMENT


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
def mobilenet_iron(use_placement=True):
    """Build the full mobilenet IRON design and return the resolved Program."""
    P = PLACEMENT if use_placement else {}

    # Runtime arg types: i32 element view over the underlying byte buffers.
    #   arg0 (act_in / scratch):  100352 i32 = 401408 bytes
    #   arg2 (final FC2 output):    640 i32 =   2560 bytes
    in_ty = np.ndarray[(tensorInW * tensorInH * tensorInC // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[
        (post_L1_OutW * post_L1_OutH * post_L2_OutC * 2 // 4,),
        np.dtype[np.int32],
    ]

    # ------------------------------------------------------------------
    # Block chain — each builder owns its fifos / kernels / workers and
    # returns the activation handoff for the next stage.
    # ------------------------------------------------------------------
    init_workers, act_in, act_init_out = init_conv(
        sf, placement=P.get("init"), data_dir=data_dir
    )
    a_workers, act_bn9_out = regular_bottlenecks(
        act_init_out, sf, placement=P.get("regular"), data_dir=data_dir
    )
    b_workers, act_bn12_out = pipeline_bottlenecks(
        act_bn9_out, sf, placement=P.get("pipeline"), data_dir=data_dir
    )
    c_workers, act_bn14_out, wts_fifos = cascade_bottlenecks(
        act_bn12_out, sf, placement=P.get("cascade"), data_dir=data_dir
    )
    l1_workers, act_out_post_avgpool_shim = post_l1(
        act_bn14_out, sf, placement=P.get("post_l1"), data_dir=data_dir
    )

    # post_l1 drains to host scratch and post_l2 fills from host scratch; this
    # bridge fifo is the runtime-sequence-side handle to that round-trip.
    # Neither builder is the natural owner — the top file glues them together.
    act_out_post_shim_FC = ObjectFifo(
        np.ndarray[(post_L2_InC,), np.dtype[np.uint16]],
        depth=2,
    )

    l2_workers, act_out_of = post_l2(
        act_out_post_shim_FC, sf, placement=P.get("post_l2"), data_dir=data_dir
    )

    # ------------------------------------------------------------------
    # Collect all workers
    # ------------------------------------------------------------------
    all_workers = (
        init_workers + a_workers + b_workers + c_workers + l1_workers + l2_workers
    )

    # Combined cascade weight tensor — test_mobilenet.py concatenates 4 chunks
    # into a single buffer in this exact order:
    #   bn13_L1(76800) | bn13_L3(76800) | bn14_L1(76800) | bn14_L3(76800)
    from aie.helpers.taplib import TensorAccessPattern

    _BN_L1_SZ = 80 * 960  # 76800 bytes per L1 weight chunk
    _BN_L3_SZ = 480 * 80 * 2  # 76800 bytes per L3 weight chunk (put+get)
    _CASCADE_OFFSETS = [0, _BN_L1_SZ, 2 * _BN_L1_SZ, 3 * _BN_L1_SZ]
    _CASCADE_SIZES = [_BN_L1_SZ, _BN_L3_SZ, _BN_L1_SZ, _BN_L3_SZ]
    _cascade_wts_sz_i32 = sum(_CASCADE_SIZES) // 4  # 76800 i32 = 307200 bytes
    cascade_wts_ty = np.ndarray[(_cascade_wts_sz_i32,), np.dtype[np.int32]]

    def _wts_tap(byte_offset, byte_size):
        return TensorAccessPattern(
            (_cascade_wts_sz_i32,),
            offset=byte_offset // 4,
            sizes=[1, 1, 1, byte_size // 4],
            strides=[0, 0, 0, 1],
        )

    # Use the gemm-style "one task_group at a time" pattern. Each task_group
    # holds a batch of fills + a wait=True drain, and finish_task_group()
    # awaits the drain AND frees every task in the group atomically. This
    # avoids the previous bug of `dma_free_task` being emitted for tasks
    # that were never awaited (act_in, weights, FC fills) — which deallocated
    # their BD IDs while their DMAs were potentially still in flight.
    rt = Runtime()
    with rt.sequence(in_ty, cascade_wts_ty, out_ty) as (inp, cascade_wts, out):
        rt.start(*all_workers)

        # ---- Group 1: input + weights + avgpool drain ----
        # All upstream fills + the first sync drain in the same group. By the
        # time the avgpool drain completes, init_conv has consumed all of
        # act_in and bn13/14 have consumed their weights, so freeing all of
        # those at finish_task_group() is safe (causal closure).
        shim = P.get("shim", {})
        tg1 = rt.task_group()
        rt.fill(
            act_in.prod(depth=1),
            inp,
            **tile_kw(shim.get("input")),
            task_group=tg1,
        )
        # bn13/14 L1+L3 weight chunks from the combined cascade buffer
        wts_shims = shim.get("wts", [None] * 4)
        for fifo, off, sz, s in zip(
            wts_fifos, _CASCADE_OFFSETS, _CASCADE_SIZES, wts_shims
        ):
            rt.fill(
                fifo.prod(),
                cascade_wts,
                _wts_tap(off, sz),
                **tile_kw(s),
                task_group=tg1,
            )
        # Round-trip avgpool output through L3 (shim 30/40 hop). Reuse `inp`
        # as scratch — input is fully consumed by the time PostL1 emits output.
        # Offsets/sizes are i32 elements (4 bytes each):
        #   avgpool / FC1-input scratch: i32 offset 640  (byte 2560)
        #   FC1-output / FC2-input scratch: i32 offset 1280 (byte 5120)
        #   transfer length: 640 i32 = 2560 B = 1280 ui16
        _post_l1_out_sz_i32 = post_L1_OutW * post_L1_OutH * post_L2_InC * 2 // 4  # 640
        _inp_sz_i32 = tensorInW * tensorInH * tensorInC // 4  # 100352
        _post_l1_scratch_tap = TensorAccessPattern(
            (_inp_sz_i32,),
            offset=_post_l1_out_sz_i32,
            sizes=[1, 1, 1, _post_l1_out_sz_i32],
            strides=[0, 0, 0, 1],
        )
        rt.drain(
            act_out_post_avgpool_shim.cons(),
            inp,
            tap=_post_l1_scratch_tap,
            wait=True,
            task_group=tg1,
            **tile_kw(shim.get("scratch_drain")),
        )
        rt.finish_task_group(tg1)

        # ---- Group 2: FC1 fill + FC1 drain ----
        # FC1 fill reads the avgpool scratch (drained above). FC1 drain
        # waits for FC compute to consume the fill, then drains FC1 output
        # to L3. By finish_task_group, both FC1 fill and FC1 drain have
        # completed.
        tg2 = rt.task_group()
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_l1_scratch_tap,
            task_group=tg2,
            **tile_kw(shim.get("fc_fill")),
        )
        _post_fc_out_tap = TensorAccessPattern(
            (_inp_sz_i32,),
            offset=_post_l1_out_sz_i32 * 2,  # i32 offset 1280
            sizes=[1, 1, 1, _post_l1_out_sz_i32],
            strides=[0, 0, 0, 1],
        )
        rt.drain(
            act_out_of.cons(),
            inp,
            tap=_post_fc_out_tap,
            wait=True,
            task_group=tg2,
            **tile_kw(shim.get("fc_drain")),
        )
        rt.finish_task_group(tg2)

        # ---- Group 3: FC2 fill + FC2 final drain to host ----
        tg3 = rt.task_group()
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_fc_out_tap,
            task_group=tg3,
            **tile_kw(shim.get("fc_fill")),
        )
        rt.drain(
            act_out_of.cons(),
            out,
            wait=True,
            task_group=tg3,
            **tile_kw(shim.get("fc_drain")),
        )
        rt.finish_task_group(tg3)

    # ------------------------------------------------------------------
    # Generate MLIR
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate MobileNet V3 IRON MLIR.")
    ap.add_argument(
        "--no-placement",
        action="store_true",
        help="Emit unplaced MLIR (for SA placer). Default: use placement.py hints.",
    )
    args = ap.parse_args()
    print(mobilenet_iron(use_placement=not args.no_placement))
