#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One bottleneck family of the IRON mobilenet design, built on its own.

The full network (aie2_mobilenet_iron.py) is hard to debug when it disagrees
with the golden output, since the error could come from any of 15 blocks. This
builds just one family's consecutive blocks, fed and drained by the host, so
each can be checked bit-exact against the brevitas reference for that family
alone (the per-chain fixtures in bottleneck_{A,B,C}/data/):

    regular    - bn0 -> bn9    (regular_bottlenecks; bn0 input is uint8)
    pipeline   - bn10 -> bn12  (pipeline_bottlenecks)
    cascade    - bn13 -> bn14  (cascade_bottlenecks)

It uses the same block builders as the full design, so it is a test harness
for them rather than a design or a transform of its own. test_e2e.py runs it
on hardware; this module's own entry point prints the MLIR, or compiles it
given --xclbin-path/--insts-path. Run `python3 -m mobilenet.aie2_iron_chain
--help` from programming_examples/ml for the options.
"""

import argparse
import json

import aie.iron as iron
import numpy as np
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


def build_chain(
    mode: CompileTime[str], data_dir: CompileTime[str], scales_json: CompileTime[str]
):
    """The resolved MLIR module for one family's blocks, `mode` naming the
    family ('regular', 'pipeline' or 'cascade'): a host-filled input
    ObjectFifo, the family's workers, and a host-drained output ObjectFifo.
    """
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
        workers, act_out = regular_bottlenecks(act_in, sf, data_dir=data_dir)
    elif mode == "pipeline":
        workers, act_out = pipeline_bottlenecks(act_in, sf, data_dir=data_dir)
    else:  # cascade
        workers, act_out = cascade_bottlenecks(act_in, sf, data_dir=data_dir)

    def sequence(inp, out, in_prod, out_cons):
        tg = TaskGroup()
        in_prod.fill(inp, group=tg)
        out_cons.drain(out, wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
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
    return build_chain(mode, data_dir, scales_json)


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="python3 -m mobilenet.aie2_iron_chain",
        description="Build one bottleneck family of the IRON mobilenet design "
        "on its own and print its MLIR, or compile it with "
        "--xclbin-path/--insts-path.",
        epilog="examples (from programming_examples/ml):\n"
        "  python3 -m mobilenet.aie2_iron_chain regular "
        "--data-dir mobilenet/bottleneck_A/data \\\n"
        "      --scales-json mobilenet/bottleneck_A/data/scale_factors_fused.json\n"
        "  python3 -m mobilenet.aie2_iron_chain pipeline "
        "--data-dir mobilenet/bottleneck_B/data \\\n"
        "      --scales-json mobilenet/bottleneck_B/data/scale_factors.json\n"
        "  python3 -m mobilenet.aie2_iron_chain cascade "
        "--data-dir mobilenet/bottleneck_C/data \\\n"
        "      --scales-json mobilenet/bottleneck_C/data/scale_factors.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument(
        "mode",
        choices=["regular", "pipeline", "cascade"],
        help="blocks to build: bn0-bn9, bn10-bn12 or bn13-bn14",
    )
    p.add_argument("--data-dir", required=True, help="weights directory")
    p.add_argument("--scales-json", required=True, help="scale_factors JSON path")
    p.add_argument(
        "--sa-effort",
        type=float,
        help="SA placer search budget scale (default: 1.0; lower trades "
        "placement cost for compile time)",
    )
    return p


def main():
    opts = _make_argparser().parse_args()
    set_current_device(device_from_args(opts, n_cols=None))
    design = chain_design
    if opts.sa_effort is not None:
        design = design.specialize(aiecc_flags=sa_placer_flags(effort=opts.sa_effort))
    compile_kwargs = dict(
        mode=opts.mode, data_dir=opts.data_dir, scales_json=opts.scales_json
    )
    if opts.xclbin_path:
        design.specialize(**compile_kwargs).compile(
            xclbin_path=opts.xclbin_path, inst_path=opts.insts_path
        )
    else:
        print(build_chain(**compile_kwargs))


if __name__ == "__main__":
    main()
