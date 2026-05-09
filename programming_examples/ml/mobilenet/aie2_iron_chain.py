#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Build + emit MLIR for a chained subset of the IRON mobilenet design.

Two preset chains are supported, mirroring the bottleneck_B / bottleneck_C
brevitas reference designs:

    pipeline   - bn10 -> bn11 -> bn12  (uses pipeline_bottlenecks)
    cascade    - bn13 -> bn14          (uses cascade_bottlenecks)

These match the per-chain golden fixtures in bottleneck_{B,C}/data/ so a
hardware run can be compared bit-exact against brevitas.

Usage:
    python3 aie2_iron_chain.py pipeline  --data-dir bottleneck_B/data \\
        --scales-json bottleneck_B/data/scale_factors.json > chain.mlir
    python3 aie2_iron_chain.py cascade   --data-dir bottleneck_C/data \\
        --scales-json bottleneck_C/data/scale_factors.json > chain.mlir
"""

import os
import sys
import json
import argparse

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern

from network_spec import block as nsblock
from bottleneck._common import i8 as _i8
from bottleneck.pipeline import pipeline_bottlenecks
from bottleneck.cascade import cascade_bottlenecks

T = Tile

# Test placements for chain designs — single-column-ish layouts that don't
# collide with the main mobilenet's PLACEMENT (which packs every column).
CHAIN_PLACEMENT = {
    # Pipeline placement mirrors aie2_mobilenet_iron.py PLACEMENT["pipeline"] —
    # spread across multiple columns so the AIE memory allocator has room.
    "pipeline": {
        "bn10": {"l1": T(1, 5), "l2": T(2, 4), "l3": T(2, 5)},
        "bn11": {
            "l1": T(3, 2),
            "l2": T(3, 4),
            "l3": T(2, 2),
            "mem_skip": T(2, 1),
        },
        "bn12": {"l1": T(3, 5), "l23": T(4, 4)},
    },
    # Cascade placement also mirrors PLACEMENT["cascade"].
    "cascade": {
        "bn13": {
            "l1_put": T(4, 5),
            "l1_get": T(5, 5),
            "l2": T(5, 4),
            "l3_put": T(4, 3),
            "l3_get": T(5, 3),
            "mem_l1": T(0, 1),
            "mem_l3": T(1, 1),
            "mem_skip": T(5, 1),
        },
        "bn14": {
            "l1_put": T(6, 5),
            "l1_get": T(7, 5),
            "l2": T(6, 2),
            "l3_put": T(4, 2),
            "l3_get": T(5, 2),
            "mem_l1": T(2, 1),
            "mem_l3": T(3, 1),
            "mem_skip": T(7, 1),
        },
    },
    # Shim DMAs (separate tiles for input vs. output).
    "shim_input": T(0, 0),
    "shim_output": T(1, 0),
    # Cascade weight fills (bn13_l1, bn13_l3, bn14_l1, bn14_l3).
    "shim_wts": [T(c, 0) for c in (4, 5, 6, 7)],
}


def _chain_iron(mode, data_dir, scales_json):
    """Build a chained design (mode='pipeline' or 'cascade'). Returns MLIR."""
    if not data_dir.endswith("/"):
        data_dir = data_dir + "/"
    with open(scales_json) as f:
        sf = json.load(f)

    if mode == "pipeline":
        in_blk = nsblock("bn10")
        out_blk = nsblock("bn12")
    elif mode == "cascade":
        in_blk = nsblock("bn13")
        out_blk = nsblock("bn14")
    else:
        raise ValueError(f"unknown chain mode: {mode!r}")

    in_w, in_h, in_c = in_blk.layers[0].in_shape
    out_w, out_h, out_c = out_blk.layers[-1].out_shape

    # i32-flat host buffer types
    in_ty = np.ndarray[(in_w * in_h * in_c // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[(out_w * out_h * out_c // 4,), np.dtype[np.int32]]

    # Chain input fifo: int8 row-by-row.
    act_in = ObjectFifo(_i8((in_w, 1, in_c)), depth=2)

    if mode == "pipeline":
        workers, act_out = pipeline_bottlenecks(
            act_in,
            sf,
            placement=CHAIN_PLACEMENT["pipeline"],
            data_dir=data_dir,
        )
        wts_fifos = []
    else:  # cascade
        workers, act_out, wts_fifos = cascade_bottlenecks(
            act_in,
            sf,
            placement=CHAIN_PLACEMENT["cascade"],
            data_dir=data_dir,
        )

    rt = Runtime()
    if wts_fifos:
        # Cascade: input + ONE concatenated cascade weight buffer + output.
        # All 4 weight chunks live in a single host tensor; TensorAccessPatterns
        # slice it for each fifo. Mirrors aie2_mobilenet_iron.py main runtime.
        BN_WTS_SZ = 80 * 960  # 76800 bytes per chunk
        TOTAL_WTS_SZ_I32 = 4 * BN_WTS_SZ // 4  # 76800 i32 elements
        wts_ty = np.ndarray[(TOTAL_WTS_SZ_I32,), np.dtype[np.int32]]
        offsets_i32 = [
            i * (BN_WTS_SZ // 4) for i in range(4)
        ]  # [0, 19200, 38400, 57600]
        size_i32 = BN_WTS_SZ // 4  # 19200

        def _wts_tap(byte_offset_i32):
            return TensorAccessPattern(
                (TOTAL_WTS_SZ_I32,),
                offset=byte_offset_i32,
                sizes=[1, 1, 1, size_i32],
                strides=[0, 0, 0, 1],
            )

        with rt.sequence(in_ty, wts_ty, out_ty) as (inp, all_wts, out):
            rt.start(*workers)
            tg = rt.task_group()
            rt.fill(
                act_in.prod(depth=1),
                inp,
                tile=CHAIN_PLACEMENT["shim_input"],
                task_group=tg,
            )
            for fifo, off, shim in zip(
                wts_fifos, offsets_i32, CHAIN_PLACEMENT["shim_wts"]
            ):
                rt.fill(
                    fifo.prod(),
                    all_wts,
                    _wts_tap(off),
                    tile=shim,
                    task_group=tg,
                )
            rt.drain(
                act_out.cons(),
                out,
                wait=True,
                tile=CHAIN_PLACEMENT["shim_output"],
                task_group=tg,
            )
            rt.finish_task_group(tg)
    else:
        with rt.sequence(in_ty, out_ty) as (inp, out):
            rt.start(*workers)
            tg = rt.task_group()
            rt.fill(
                act_in.prod(depth=1),
                inp,
                tile=CHAIN_PLACEMENT["shim_input"],
                task_group=tg,
            )
            rt.drain(
                act_out.cons(),
                out,
                wait=True,
                tile=CHAIN_PLACEMENT["shim_output"],
                task_group=tg,
            )
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a chained IRON mobilenet subset.")
    ap.add_argument("mode", choices=["pipeline", "cascade"])
    ap.add_argument("--data-dir", required=True, help="weights directory")
    ap.add_argument("--scales-json", required=True, help="scale_factors JSON path")
    args = ap.parse_args()
    print(_chain_iron(args.mode, args.data_dir, args.scales_json))
