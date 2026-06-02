# section-1/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Section-1 minimal IRON design — higher-level (``@iron.jit``) entry point.

A single ``Worker`` placed on tile (0, 2) writes zeros into a local
``Buffer``.  The runtime sequence has one tensor argument (the host-
facing output) and starts the worker.

The whole design is just a function decorated with ``@iron.jit``: the
first call JIT-compiles + runs on the attached NPU; ``--emit-mlir``
prints the lowered MLIR without touching the hardware.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

data_size = 48
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]


@iron.jit
def section_one(b_out: Out):
    buf = Buffer(data_ty, name="buff")

    def core_fn(buff):
        for i in range_(data_size):
            buff[i] = 0

    my_worker = Worker(core_fn, [buf], tile=Tile(0, 2), while_true=False)

    rt = Runtime()
    with rt.sequence(data_ty) as _:
        rt.start(my_worker)

    return Program(iron.get_current_device(), rt).resolve_program()


def _run_and_verify(opts):
    out = iron.zeros(data_size, dtype=np.int32, device="npu")
    section_one(out)
    print("PASS!")


def _emit_mlir(opts):
    out = iron.zeros(data_size, dtype=np.int32, device="npu")
    print(section_one.as_mlir(out))


def main():
    p = argparse.ArgumentParser(prog="Section 1 minimal IRON design")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        section_one,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
