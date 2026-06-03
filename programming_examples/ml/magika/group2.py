# magika/group2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""Magika group2 — IRON API design with ``@iron.jit`` compilation.

Single-tile design with 4 LUT buffers spread across neighboring tiles
(south / west / north / current) for memory capacity. The kernel uses
``put_ms()`` to push outputs directly to the wire, so the output
ObjectFifo is constructed with ``aie_stream=(end, port)`` to mark it
as a direct-stream connection.
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, Compile, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, device_from_args
from aie.iron.kernel import ExternalFunction
from aie.dialects._aie_enum_gen import AIETileType
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli


_THIS_DIR = Path(__file__).parent
_KERNEL_SRC = _THIS_DIR / "kernels" / "group2.cc"
_KERNEL_INC = _THIS_DIR / "inc"
_DATA_DIR = _THIS_DIR / "data"


@iron.jit
def group2(
    a_in: In,
    _dummy: In,
    c_out: Out,
    *,
    trace_size: Compile[int] = 0,
):
    din_size = 16 * 43
    dout_size = 214

    din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
    dout_ty = np.ndarray[(dout_size,), np.dtype[np.int32]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    lut0_arr = np.loadtxt(_DATA_DIR / "lut0_group2.txt", delimiter=",")
    lut1_arr = np.loadtxt(_DATA_DIR / "lut1_group2.txt", delimiter=",")
    lut2_arr = np.loadtxt(_DATA_DIR / "lut2_group2.txt", delimiter=",")
    lut3_arr = np.loadtxt(_DATA_DIR / "lut3_group2.txt", delimiter=",")

    lut0_ty = np.ndarray[(lut0_arr.size,), np.dtype[np.int16]]
    lut1_ty = np.ndarray[(lut1_arr.size,), np.dtype[np.int16]]
    lut2_ty = np.ndarray[(lut2_arr.size,), np.dtype[np.int16]]
    lut3_ty = np.ndarray[(lut3_arr.size,), np.dtype[np.int16]]

    # Tile placement matches the dialect-direct original (col=1, row=3).
    # LUTs are spread across the 4 neighboring tiles for memory capacity.
    shim_tile = Tile(col=1, row=0, tile_type=AIETileType.ShimNOCTile)
    mem_tile = Tile(col=1, row=1, tile_type=AIETileType.MemTile)
    compute_tile = Tile(col=1, row=3, tile_type=AIETileType.CoreTile)
    south_tile = Tile(col=1, row=2, tile_type=AIETileType.CoreTile)
    north_tile = Tile(col=1, row=4, tile_type=AIETileType.CoreTile)
    west_tile = Tile(col=0, row=3, tile_type=AIETileType.CoreTile)

    # LUTs live on the 4 neighbor tiles (compute tile reads N/S/W neighbors'
    # L1 directly via shared memory).  Worker preserves these explicit
    # placements; it only auto-pins Buffers that were created without a tile.
    lut0_buf = Buffer(tile=west_tile, type=lut0_ty,
                      initial_value=np.array(lut0_arr, dtype=np.int16), name="lut0_buf")
    lut1_buf = Buffer(tile=north_tile, type=lut1_ty,
                      initial_value=np.array(lut1_arr, dtype=np.int16), name="lut1_buf")
    lut2_buf = Buffer(tile=south_tile, type=lut2_ty,
                      initial_value=np.array(lut2_arr, dtype=np.int16), name="lut2_buf")
    lut3_buf = Buffer(tile=compute_tile, type=lut3_ty,
                      initial_value=np.array(lut3_arr, dtype=np.int16), name="lut3_buf")

    group2_kernel = ExternalFunction(
        "group2_kernel",
        source_file=str(_KERNEL_SRC),
        include_dirs=[str(_KERNEL_INC), cxx_header_path()],
        arg_types=[din_ty, lut0_ty, lut1_ty, lut2_ty, lut3_ty],
        object_file_name="group2.o",
    )

    # Input chain: shim -> mem -> compute.
    of_din_L3L2 = ObjectFifo(din_ty, name="of_din_L3L2", depth=2)
    of_din_L2L1 = of_din_L3L2.cons().forward(name="of_din_L2L1", obj_type=din_ty)

    # Output goes compute -> shim as a direct stream (no L1 buffer); the
    # kernel emits each element via put_ms() instead of acquire/release.
    of_dout_L1L3 = ObjectFifo(
        dout_ty, name="of_dout_L1L3", depth=2, aie_stream=(0, 0)
    )

    # Core body: kernel writes its output via put_ms() inside group2.cc, so
    # we don't acquire/release of_dout here.
    def core_body(of_in, _of_out_unused, lut0, lut1, lut2, lut3, kernel):
        for _ in range_(0x7FFFFFFF):
            di = of_in.acquire(1)
            kernel(di, lut0, lut1, lut2, lut3)
            of_in.release(1)

    worker = Worker(
        core_body,
        fn_args=[
            of_din_L2L1.cons(),
            of_dout_L1L3.prod(),
            lut0_buf,
            lut1_buf,
            lut2_buf,
            lut3_buf,
            group2_kernel,
        ],
        tile=compute_tile,
        while_true=False,
    )

    rt = Runtime()
    with rt.sequence(din_ty, scalar_ty, dout_ty) as (a, _b, c):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(worker)
        rt.fill(of_din_L3L2.prod(), a, tile=shim_tile)
        rt.drain(of_dout_L1L3.cons(), c, tile=shim_tile, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Magika group2")
    add_compile_args(
        p, dev_choices=("npu", "npu2"), default_dev="npu", with_emit_mlir=True
    )
    add_trace_arg(p)
    return p


def _compile_kwargs(opts):
    return dict(trace_size=opts.trace_size)


def _emit_mlir(opts):
    print(group2.as_mlir(None, None, None, **_compile_kwargs(opts)))


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        group2,
        opts,
        compile_kwargs=_compile_kwargs,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=2),
    )


if __name__ == "__main__":
    main()
