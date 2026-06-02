# distribute_L2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""``ObjectFifo.cons().split(...)`` distribute-from-L2 structure — ``@iron.jit``.

A structural-only example: the shim brings a 24-element int32 vector
into a mem tile, which fans out 8-element sub-tiles to ``n_workers``
compute tiles via ``of_in.cons().split(...)``.  Each worker just
acquires + releases its tile (the data path is the point — there is no
compute).  Useful for inspecting the generated MLIR for the distribute
pattern.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

n_workers = 3

data_ty = np.ndarray[(48,), np.dtype[np.int32]]
tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]


@iron.jit
def distribute_L2(a_in: In):
    of_offsets = [8 * w for w in range(n_workers)]

    of_in = ObjectFifo(tile24_ty, name="in")
    of_ins = of_in.cons().split(
        of_offsets,
        obj_types=[tile8_ty] * n_workers,
        names=[f"in{w}" for w in range(n_workers)],
    )

    def core_fn(of_in):
        elem = of_in.acquire(1)
        of_in.release(1)

    workers = [Worker(core_fn, [of_ins[w].cons()]) for w in range(n_workers)]

    rt = Runtime()
    with rt.sequence(data_ty) as a:
        rt.start(*workers)
        rt.fill(of_in.prod(), a)

    return Program(iron.get_current_device(), rt).resolve_program()


def _emit_mlir(opts):
    a_in = iron.zeros(48, dtype=np.int32, device="npu")
    print(distribute_L2.as_mlir(a_in))


def main():
    p = argparse.ArgumentParser(prog="distribute (L2 split) structural example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    # No run_and_verify: the design has no compute, just structural fan-out.
    # --emit-mlir is the intended use.
    if not getattr(opts, "emit_mlir", False):
        opts.emit_mlir = True
    run_design_cli(
        distribute_L2, opts, compile_kwargs={},
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
