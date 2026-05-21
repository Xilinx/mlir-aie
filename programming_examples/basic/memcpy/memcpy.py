# memcpy/memcpy.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Memcpy microbenchmark — Iron API design with ``@iron.jit`` compilation.

Saturates DDR bandwidth by using every column's shim DMA in-out pairs.
Two paths:

  * ``--bypass``: shim → memtile → shim via ObjectFifo.forward() (no
    compute tile).
  * Compute path: per-column / per-channel passthrough kernel
    (``kernels.passthrough``) on a compute tile.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tensortiler2d import TensorTiler2D
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    if dev_str == "npu":
        return NPU1()
    if dev_str == "npu2":
        return NPU2()
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


@iron.jit
def memcpy(
    a_in: In,
    b_out: Out,
    *,
    size: Compile[int] = 16384,
    num_columns: Compile[int] = 1,
    num_channels: Compile[int] = 1,
    bypass: Compile[bool] = True,
):
    xfr_dtype = np.int32
    line_size = 1024
    line_type = np.ndarray[(line_size,), np.dtype[xfr_dtype]]
    transfer_type = np.ndarray[(size,), np.dtype[xfr_dtype]]

    chunk = size // num_columns // num_channels

    of_ins = [
        ObjectFifo(line_type, name=f"in{i}_{j}")
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    if bypass:
        # Pure shim→memtile→shim forward; no Worker needed.
        of_outs = [
            of_ins[i * num_channels + j].cons().forward()
            for i in range(num_columns)
            for j in range(num_channels)
        ]
        my_workers = []
    else:
        of_outs = [
            ObjectFifo(line_type, name=f"out{i}_{j}")
            for i in range(num_columns)
            for j in range(num_channels)
        ]

        passthrough_fn = kernels.passthrough(tile_size=line_size, dtype=xfr_dtype)

        def core_fn(of_in, of_out, passThroughLine):
            elemOut = of_out.acquire(1)
            elemIn = of_in.acquire(1)
            passThroughLine(elemIn, elemOut, line_size)
            of_in.release(1)
            of_out.release(1)

        my_workers = [
            Worker(
                core_fn,
                [
                    of_ins[i * num_channels + j].cons(),
                    of_outs[i * num_channels + j].prod(),
                    passthrough_fn,
                ],
            )
            for i in range(num_columns)
            for j in range(num_channels)
        ]

    # One TAP per (column, channel) shim DMA — same as iterating
    # `(1, chunk)` tiles row-major across the `(1, size)` tensor.
    taps = TensorTiler2D.simple_tiler((1, size), (1, chunk))

    rt = Runtime()
    with rt.sequence(transfer_type, transfer_type) as (a, b):
        if my_workers:
            rt.start(*my_workers)
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_ins[i * num_channels + j].prod(),
                    a,
                    taps[i * num_channels + j],
                )
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    b,
                    taps[i * num_channels + j],
                    wait=True,
                )

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Memcpy")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-l", "--length", type=int, default=16384, help="transfer size")
    p.add_argument("-co", "--cols", type=int, default=1, help="number of columns")
    p.add_argument("-ch", "--chans", type=int, default=1, help="channels per column (1 or 2)")
    p.add_argument(
        "-b",
        "--bypass",
        type=str,
        default="True",
        help="use the DMA-only forward path (yes/true/t/1 → True)",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    p.add_argument(
        "--elf-path",
        type=str,
        default=None,
        help="optional ELF-wrapped insts (for the test.cpp xrt::elf flow)",
    )
    return p


def _validate(opts):
    max_cols = 4 if opts.dev == "npu" else 8
    if opts.cols > max_cols:
        sys.exit(f"--cols ({opts.cols}) exceeds {opts.dev} max ({max_cols})")
    if opts.chans not in (1, 2):
        sys.exit(f"--chans must be 1 or 2, got {opts.chans}")
    if (opts.length % 1024) % opts.cols % opts.chans != 0:
        sys.exit(
            f"--length ({opts.length}) must be a multiple of 1024 and divisible by "
            f"--cols × --chans ({opts.cols} × {opts.chans})"
        )


def _bypass_bool(s: str) -> bool:
    return s.lower() in ("yes", "true", "t", "1")


def _compile_kwargs(opts):
    return dict(
        size=opts.length,
        num_columns=opts.cols,
        num_channels=opts.chans,
        bypass=_bypass_bool(opts.bypass),
    )


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = memcpy.specialize(**_compile_kwargs(opts))
    spec.compile(
        xclbin_path=opts.xclbin_path,
        inst_path=opts.insts_path,
        elf_path=opts.elf_path,
    )


def _run_and_verify(opts):
    in_np = np.arange(opts.length, dtype=np.int32)
    out_np = np.zeros_like(in_np)

    a_t = iron.tensor(in_np, dtype=np.int32, device="npu")
    b_t = iron.tensor(out_np, dtype=np.int32, device="npu")

    memcpy(a_t, b_t, **_compile_kwargs(opts))

    if not np.array_equal(b_t.numpy(), in_np):
        sys.exit("FAIL! output does not match input")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    _validate(opts)
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
