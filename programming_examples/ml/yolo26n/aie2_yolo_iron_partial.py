# aie2_yolo_iron_partial.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
#
"""Partial-chain IRON design — builds any contiguous span of blocks.

By default chains m0..m10 (full network); override with the environment
variable CHAIN_BLOCKS, e.g. `CHAIN_BLOCKS=m0,m1,m2` to bring up a
sub-chain incrementally. The full chain is bit-exact NPU==ORT.

Per-block kernel symbol mangling lets blocks sharing a kernel template
(c3k2_small for m2/m4, c3k2_heavy for m6/m8, chunked conv_stride for
m3/m5/m7) coexist in one MLIR module: each block compiles its .cc with
-DKERNEL_SUFFIX=_mN producing a uniquely-named .o.

Usage:
    python3 aie2_yolo_iron_partial.py > build/aie_chain.mlir
    CHAIN_BLOCKS=m0,m1 python3 aie2_yolo_iron_partial.py > build/aie_chain.mlir
"""

from __future__ import annotations

import os
import sys
import pathlib

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import yolo_spec
import placement
from aie2_yolo_per_block import (
    _BUILDERS,
    _load_manifest,
    _i8,
    _i32,
    TRACE_SIZE_PER_WORKER,
    TRACE_EVENTS,
)

_DEFAULT_BLOCKS = ("m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10")
# CHAIN_BLOCKS env override for bisect-style debug:
#   CHAIN_BLOCKS=m0   → just m0
#   CHAIN_BLOCKS=m0,m1 → m0 then m1
_chain_env = os.environ.get("CHAIN_BLOCKS", "").strip()
CHAIN_BLOCKS = tuple(_chain_env.split(",")) if _chain_env else _DEFAULT_BLOCKS

# Number of samples per dispatch. Set N_SAMPLES > 1 to size the rt.fill /
# rt.drain transfers for batched inference — the workers' outer infinite
# scf.for loop consumes whatever upstream produces, so the pipeline
# naturally streams N samples back-to-back, and we can measure steady-
# state throughput (fps = N / total_dispatch_time).
import os

N_SAMPLES = int(os.environ.get("CHAIN_N_SAMPLES", "1"))


def yolo_iron_partial() -> str:
    manifest = _load_manifest()

    m0 = yolo_spec.block("m0")
    in_w, in_h, _ = m0.layers[0].in_shape
    IN_C_PADDED = 8
    in_bytes = in_w * in_h * IN_C_PADDED
    in_ty = _i32((N_SAMPLES * in_bytes // 4,))

    last = yolo_spec.block(CHAIN_BLOCKS[-1])
    last_shape = last.layers[-1].out_shape
    # m10 head emits a flat (out_c,) probs vector; spatial blocks emit (W,H,C).
    out_bytes_per = int(np.prod(last_shape))
    # m10 output padded to 4 bytes for shim alignment.
    if last_shape == (last.layers[-1].out_shape) and last.topology == "head":
        out_bytes_per = max(4, ((out_bytes_per + 3) // 4) * 4)
    out_ty = _i32((N_SAMPLES * ((out_bytes_per + 3) // 4),))

    act_in_fifo = ObjectFifo(_i8((in_w, 1, IN_C_PADDED)), depth=5)

    current_fifo = act_in_fifo
    all_workers = []
    workers_by_block: dict[str, list] = {}
    for name in CHAIN_BLOCKS:
        out_fifo, workers = _BUILDERS[name](current_fifo, manifest)
        all_workers.extend(workers)
        workers_by_block[name] = list(workers)
        current_fifo = out_fifo

    # TRACE_BLOCKS = comma-separated subset of CHAIN_BLOCKS whose workers
    # get included in the trace. Required because the chain (~32 workers)
    # exceeds the upstream 31-tile packet-trace flow cap; a two-pass trace
    # (front half + back half) is the workaround. Default = all blocks.
    _trace_env = os.environ.get("TRACE_BLOCKS", "").strip()
    trace_blocks = (
        tuple(b for b in _trace_env.split(",") if b)
        if _trace_env
        else tuple(CHAIN_BLOCKS)
    )
    workers_to_trace = [w for b in trace_blocks for w in workers_by_block.get(b, [])]

    in_i32_per_sample = in_bytes // 4
    out_i32_per_sample = (out_bytes_per + 3) // 4
    in_total_i32 = N_SAMPLES * in_i32_per_sample
    out_total_i32 = N_SAMPLES * out_i32_per_sample

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (inp, out):
        if TRACE_SIZE_PER_WORKER > 0 and workers_to_trace:
            # Append trace BO after the last sequence tensor (`out`); host
            # runtime resizes `out` to include trace bytes when ddr_id == -1.
            # Total trace BO bytes = TRACE_SIZE_PER_WORKER * len(workers_to_trace);
            # bench_block/bench_chain auto-detect this from the lowered MLIR.
            # 31-tile cap (feedback_trace_31_tile_cap.md) limits per build.
            assert len(workers_to_trace) <= 31, (
                f"trace requested for {len(workers_to_trace)} workers > 31-tile cap; "
                f"split via TRACE_BLOCKS"
            )
            # ddr_id default 4 (separate group_id-7 BO). ddr_id=-1 (append
            # to args[-1]) clobbers m10's chain output BO on shim col 7 —
            # samples come back all zeros. ddr_id=4 = dedicated BO, bit-
            # exact preserved. See feedback_chain_trace_breaks_pipeline.md.
            _ddr_id = int(os.environ.get("TRACE_DDR_ID", "4"))
            _events_kwargs = {}
            if TRACE_EVENTS is not None:
                # Pad to exactly 8 events (configure_trace requires 8); fill
                # remaining slots with the first event so the unused slots
                # don't fire spurious packets.
                evs = list(TRACE_EVENTS)
                from aie.utils.trace.events import CoreEventAIE2P

                while len(evs) < 8:
                    evs.append(CoreEventAIE2P.NONE)
                _events_kwargs["coretile_events"] = evs[:8]
            rt.enable_trace(
                trace_size=TRACE_SIZE_PER_WORKER * len(workers_to_trace),
                workers=workers_to_trace,
                ddr_id=_ddr_id,
                **_events_kwargs,
            )
        rt.start(*all_workers)
        if N_SAMPLES == 1:
            # Single-sample fast path: one DMA in, one DMA out.
            tg = rt.task_group()
            rt.fill(
                act_in_fifo.prod(),
                inp,
                tile=placement.PLACEMENT["shim"]["input"],
                task_group=tg,
            )
            rt.drain(
                current_fifo.cons(),
                out,
                wait=True,
                tile=placement.PLACEMENT["shim"]["output"],
                task_group=tg,
            )
            rt.finish_task_group(tg)
        else:
            # Multi-sample dispatch: ONE configure_task per direction with
            # HW BD-chain replay via the outermost dim acting as
            # repeat_count (see AIEDmaToNpu.cpp:393 — dim 3's size is read
            # as repeat_count). Pattern: sizes=[N, 1, 1, per_sample],
            # strides=[per_sample, 0, 0, 1]. This sidesteps the shim
            # per-channel task queue depth limit (~6 outstanding) that
            # silently deadlocks N>=7 with per-sample tasks.
            # HW caps repeat_count at 255 (8-bit field in cmd word).
            assert (
                N_SAMPLES <= 255
            ), f"N_SAMPLES={N_SAMPLES} exceeds shim repeat_count cap 255"
            tg = rt.task_group()
            in_tap = TensorAccessPattern(
                (in_total_i32,),
                offset=0,
                sizes=[N_SAMPLES, 1, 1, in_i32_per_sample],
                strides=[in_i32_per_sample, 0, 0, 1],
            )
            out_tap = TensorAccessPattern(
                (out_total_i32,),
                offset=0,
                sizes=[N_SAMPLES, 1, 1, out_i32_per_sample],
                strides=[out_i32_per_sample, 0, 0, 1],
            )
            rt.fill(
                act_in_fifo.prod(),
                inp,
                tap=in_tap,
                tile=placement.PLACEMENT["shim"]["input"],
                task_group=tg,
            )
            rt.drain(
                current_fifo.cons(),
                out,
                tap=out_tap,
                wait=True,
                tile=placement.PLACEMENT["shim"]["output"],
                task_group=tg,
            )
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    print(yolo_iron_partial())
