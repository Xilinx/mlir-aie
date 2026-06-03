# magika/group0.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
"""Magika group0 — IRON API design with ``@iron.jit`` compilation.

Two-core pipeline (group0a → group0b) on one column.  Each core has its
own LUT preloaded into local memory.  group0a takes an int16 input
sub-vector and emits 32 output sub-vectors per input via a (4, 8) loop
over scalar (xid, cid) indices passed to the kernel; group0b consumes
those and runs a 32-iter LUT-pair lookup to produce the final output.

The .cc kernel is compiled once per worker with ``-DGROUPA`` / ``-DGROUPB``
to select which entry point gets exported, matching the placed-dialect
build.  The Makefile flips between this iron design and ``group0_placed.py``
via ``use_placed=1`` (default placed).
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.iron import Buffer, Compile, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import device_from_args
from aie.iron.kernel import ExternalFunction
from aie.utils.config import cxx_header_path
from aie.utils.hostruntime.argparse import add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli


_THIS_DIR = Path(__file__).parent
_KERNEL_SRC = _THIS_DIR / "kernels" / "group0.cc"
_KERNEL_INC = _THIS_DIR / "inc"
_DATA_DIR = _THIS_DIR / "data"


@iron.jit
def group0(
    a_in: In,
    c_out: Out,
    _unused: In,
    *,
    trace_size: Compile[int] = 0,
):
    din_size = 2048
    dout_size = 4096
    data_int_size = 4096

    din_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
    dout_ty = np.ndarray[(dout_size,), np.dtype[np.int16]]
    data_int_ty = np.ndarray[(data_int_size,), np.dtype[np.int16]]
    transfer_in_ty = np.ndarray[(din_size,), np.dtype[np.int16]]
    # group0b emits 32 dout chunks per single input chunk; host buffer is one
    # contiguous 32-output tile.
    transfer_out_ty = np.ndarray[(dout_size * 32,), np.dtype[np.int16]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    # ----- group0a: scalar-indexed LUT lookup, 4x8 calls per input ---------
    lut0a_arr = np.loadtxt(_DATA_DIR / "lut0a_group0.txt", delimiter=",").astype(
        np.int16
    )
    lut0a_ty = np.ndarray[(lut0a_arr.size,), np.dtype[np.int16]]
    lut0a_buf = Buffer(lut0a_ty, initial_value=lut0a_arr, name="lut0a_buf")

    group0a_kernel = ExternalFunction(
        "group0a_kernel",
        source_file=str(_KERNEL_SRC),
        compile_flags=["-DGROUPA"],
        include_dirs=[str(_KERNEL_INC), cxx_header_path()],
        arg_types=[din_ty, data_int_ty, lut0a_ty, np.int32, np.int32],
        object_file_name="group0a.o",
    )

    # ----- group0b: 32-iter LUT-pair lookup ---------------------------------
    lut0b_a_arr = np.loadtxt(_DATA_DIR / "lut0b_a_group0.txt", delimiter=",").astype(
        np.int16
    )
    lut0b_b_arr = np.loadtxt(_DATA_DIR / "lut0b_b_group0.txt", delimiter=",").astype(
        np.int16
    )
    lut0b_a_ty = np.ndarray[(lut0b_a_arr.size,), np.dtype[np.int16]]
    lut0b_b_ty = np.ndarray[(lut0b_b_arr.size,), np.dtype[np.int16]]
    lut0b_a_buf = Buffer(lut0b_a_ty, initial_value=lut0b_a_arr, name="lut0b_a_buf")
    lut0b_b_buf = Buffer(lut0b_b_ty, initial_value=lut0b_b_arr, name="lut0b_b_buf")

    group0b_kernel = ExternalFunction(
        "group0b_kernel",
        source_file=str(_KERNEL_SRC),
        compile_flags=["-DGROUPB"],
        include_dirs=[str(_KERNEL_INC), cxx_header_path()],
        arg_types=[data_int_ty, dout_ty, lut0b_a_ty, lut0b_b_ty],
        object_file_name="group0b.o",
    )

    # ----- ObjectFifos ------------------------------------------------------
    of_din_L3L2 = ObjectFifo(transfer_in_ty, name="of_din_L3L2", depth=2)
    of_din_L2L1 = of_din_L3L2.cons().forward(name="of_din_L2L1", obj_type=din_ty)
    of_int = ObjectFifo(data_int_ty, name="of_int", depth=2)
    of_dout_L1L2 = ObjectFifo(dout_ty, name="of_dout_L1L2", depth=2)
    of_dout_L2L3 = of_dout_L1L2.cons().forward(name="of_dout_L2L3")

    # ----- Workers ----------------------------------------------------------
    def group0a_body(of_di, of_do, lut, kernel):
        di = of_di.acquire(1)
        for xid in range_(4):
            xid_i32 = arith.index_cast(xid, to=np_dtype_to_mlir_type(np.int32))
            for cid in range_(8):  # 64 / 8
                cid_i32 = arith.index_cast(cid, to=np_dtype_to_mlir_type(np.int32))
                do = of_do.acquire(1)
                kernel(di, do, lut, xid_i32, cid_i32)
                of_do.release(1)
        of_di.release(1)

    group0a_worker = Worker(
        group0a_body,
        fn_args=[of_din_L2L1.cons(), of_int.prod(), lut0a_buf, group0a_kernel],
        stack_size=4096,
    )

    def group0b_body(of_di, of_do, lut_a, lut_b, kernel):
        for _ in range_(32):  # 256 / 8
            do = of_do.acquire(1)
            di = of_di.acquire(1)
            kernel(di, do, lut_a, lut_b)
            of_di.release(1)
            of_do.release(1)

    group0b_worker = Worker(
        group0b_body,
        fn_args=[
            of_int.cons(),
            of_dout_L1L2.prod(),
            lut0b_a_buf,
            lut0b_b_buf,
            group0b_kernel,
        ],
    )

    # ----- Runtime ----------------------------------------------------------
    rt = Runtime()
    with rt.sequence(transfer_in_ty, transfer_out_ty, scalar_ty) as (A, C, _):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(group0a_worker, group0b_worker)
        rt.fill(of_din_L3L2.prod(), A)
        rt.drain(of_dout_L2L3.cons(), C, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Magika group0")
    add_compile_args(
        p, dev_choices=("npu", "npu2"), default_dev="npu", with_emit_mlir=True
    )
    add_trace_arg(p)
    return p


def _compile_kwargs(opts):
    return dict(trace_size=opts.trace_size)


def _emit_mlir(opts):
    print(group0.as_mlir(None, None, None, **_compile_kwargs(opts)))


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        group0,
        opts,
        compile_kwargs=_compile_kwargs,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
