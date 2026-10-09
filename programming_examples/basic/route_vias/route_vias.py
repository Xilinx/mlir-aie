# route_vias/route_vias.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import argparse

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    WireBundle,
)
from aie.iron import (
    Buffer,
    Flow,
    In,
    Out,
    Program,
    Runtime,
)
from aie.iron.device import Tile
from aie.utils import NPUKernel
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

N = 1024


@iron.jit
def route_vias(a_in: In, c_out: Out):
    vector_ty = np.ndarray[(N,), np.dtype[np.int32]]

    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    core = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)

    into = Flow(shim, core, src_channel=0, dst_channel=0, name="into")
    out = Flow(
        core,
        shim,
        src_channel=0,
        dst_channel=0,
        name="out",
        vias=[
            (core, (WireBundle.DMA, 0), (WireBundle.South, 0)),
            (mem, (WireBundle.North, 0), (WireBundle.South, 0)),
            (shim, (WireBundle.North, 0), (WireBundle.DMA, 0)),
        ],
    )

    buffer = Buffer(type=vector_ty, tile=core, name="buffer")

    def sequence(a, c):
        into.fill(a)
        load = into.endpoint(core).task(buffer, wait=True).start()
        load.await_()
        load.free()
        out.endpoint(core).task(buffer).start().free()
        out.drain(c, wait=True)

    runtime = Runtime(sequence, [vector_ty, vector_ty])
    runtime.add_flow(into)
    runtime.add_flow(out)
    return Program(iron.get_current_device(), runtime).resolve_program()


def _make_argparser():
    parser = argparse.ArgumentParser(prog="AIE Route Vias")
    add_compile_args(parser, dev_choices=("npu2",), with_emit_mlir=True)
    parser.add_argument(
        "--run-xclbin",
        type=str,
        help="load and run this xclbin (pairs with --run-insts)",
    )
    parser.add_argument(
        "--run-insts",
        type=str,
        help="run this instruction binary (pairs with --run-xclbin)",
    )
    return parser


def _run_and_verify(opts):
    input_tensor = iron.arange(N, dtype=np.int32, device="npu")
    output_tensor = iron.zeros_like(input_tensor)
    if opts.run_xclbin:
        NPUKernel(opts.run_xclbin, opts.run_insts)(input_tensor, output_tensor)
    else:
        route_vias(input_tensor, output_tensor)
    assert_pass(output_tensor.numpy(), input_tensor.numpy())


def _compile_kwargs(_opts):
    return {}


def _validate(opts):
    if bool(opts.run_xclbin) != bool(opts.run_insts):
        raise SystemExit("--run-xclbin and --run-insts must be set together")


if __name__ == "__main__":
    opts = _make_argparser().parse_args()
    run_design_cli(
        route_vias,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
        validate=_validate,
    )
