#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Build + emit MLIR for a chained subset of the IRON mobilenet design.

Three preset chains are supported, mirroring the bottleneck_A / B / C brevitas
reference designs:

    regular    - bn0 -> bn9    (uses regular_bottlenecks; bn0 input is uint8)
    pipeline   - bn10 -> bn12  (uses pipeline_bottlenecks)
    cascade    - bn13 -> bn14  (uses cascade_bottlenecks)

These match the per-chain golden fixtures in bottleneck_{A,B,C}/data/ so a
hardware run can be compared bit-exact against brevitas.

Usage:
    python3 aie2_iron_chain.py regular   --data-dir bottleneck_A/data \\
        --scales-json bottleneck_A/data/scale_factors_chain.json > chain.mlir
    python3 aie2_iron_chain.py pipeline  --data-dir bottleneck_B/data \\
        --scales-json bottleneck_B/data/scale_factors.json       > chain.mlir
    python3 aie2_iron_chain.py cascade   --data-dir bottleneck_C/data \\
        --scales-json bottleneck_C/data/scale_factors.json       > chain.mlir

Pass --xclbin-path/--insts-path to compile instead, kernels included.
"""

import argparse
import json

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    InOut,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
)
from aie.utils.hostruntime import set_current_device
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args

from .bottleneck._common import i8 as _i8
from .bottleneck._common import sa_placer_flags
from .bottleneck._common import u8 as _u8
from .bottleneck.cascade import cascade_bottlenecks
from .bottleneck.pipeline import pipeline_bottlenecks
from .bottleneck.regular import regular_bottlenecks
from .network_spec import block as nsblock


def _chain_iron(
    mode: CompileTime[str], data_dir: CompileTime[str], scales_json: CompileTime[str]
):
    """Build a chained design (mode='pipeline' or 'cascade'). Returns MLIR."""
    if not data_dir.endswith("/"):
        data_dir = data_dir + "/"
    with open(scales_json) as f:
        sf = json.load(f)

    if mode == "regular":
        in_blk, out_blk = nsblock("bn0"), nsblock("bn9")
    elif mode == "pipeline":
        in_blk, out_blk = nsblock("bn10"), nsblock("bn12")
    elif mode == "cascade":
        in_blk, out_blk = nsblock("bn13"), nsblock("bn14")
    else:
        raise ValueError(f"unknown chain mode: {mode!r}")

    in_w, in_h, in_c = in_blk.layers[0].in_shape
    out_w, out_h, out_c = out_blk.layers[-1].out_shape

    # i32-flat host buffer types
    in_ty = np.ndarray[(in_w * in_h * in_c // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[(out_w * out_h * out_c // 4,), np.dtype[np.int32]]

    # Chain input fifo: bn0 reads uint8 (init output); other chains read int8.
    in_elem_ty = _u8 if mode == "regular" else _i8
    act_in = ObjectFifo(in_elem_ty((in_w, 1, in_c)), depth=2)

    if mode == "regular":
        workers, act_out = regular_bottlenecks(
            act_in,
            sf,
            data_dir=data_dir,
        )
        wts_fifos = []
    elif mode == "pipeline":
        workers, act_out = pipeline_bottlenecks(
            act_in,
            sf,
            data_dir=data_dir,
        )
        wts_fifos = []
    else:  # cascade
        workers, act_out, wts_fifos = cascade_bottlenecks(
            act_in,
            sf,
            data_dir=data_dir,
        )

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

        def sequence_with_wts(inp, all_wts, out, in_prod, wts_prods, out_cons):
            tg = TaskGroup()
            in_prod.fill(inp, group=tg)
            for wts_prod, off in zip(wts_prods, offsets_i32):
                wts_prod.fill(all_wts, _wts_tap(off), group=tg)
            out_cons.drain(out, wait=True, group=tg)
            tg.finish()

        rt = Runtime(
            sequence_with_wts,
            [
                in_ty,
                wts_ty,
                out_ty,
                act_in.prod(depth=1),
                [fifo.prod() for fifo in wts_fifos],
                act_out.cons(),
            ],
        )
    else:

        def sequence_no_wts(inp, out, in_prod, out_cons):
            tg = TaskGroup()
            in_prod.fill(inp, group=tg)
            out_cons.drain(out, wait=True, group=tg)
            tg.finish()

        rt = Runtime(
            sequence_no_wts,
            [
                in_ty,
                out_ty,
                act_in.prod(depth=1),
                act_out.cons(),
            ],
        )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=sa_placer_flags())
def chain_design(
    *buffers: InOut,
    mode: CompileTime[str],
    data_dir: CompileTime[str],
    scales_json: CompileTime[str],
):
    return _chain_iron(mode, data_dir, scales_json)


def _make_argparser():
    p = argparse.ArgumentParser(description="Build a chained IRON mobilenet subset.")
    add_compile_args(p, default_dev="npu2")
    p.add_argument("mode", choices=["regular", "pipeline", "cascade"])
    p.add_argument("--data-dir", required=True, help="weights directory")
    p.add_argument("--scales-json", required=True, help="scale_factors JSON path")
    return p


def main():
    opts = _make_argparser().parse_args()
    set_current_device(device_from_args(opts, n_cols=None))
    compile_kwargs = dict(
        mode=opts.mode, data_dir=opts.data_dir, scales_json=opts.scales_json
    )
    if opts.xclbin_path:
        chain_design.specialize(**compile_kwargs).compile(
            xclbin_path=opts.xclbin_path, inst_path=opts.insts_path
        )
    else:
        print(_chain_iron(**compile_kwargs))


if __name__ == "__main__":
    main()
