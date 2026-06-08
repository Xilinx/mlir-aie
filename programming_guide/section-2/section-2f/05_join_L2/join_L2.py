# join_L2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""``ObjectFifo.prod().join(...)`` join-to-L2 structure — ``@iron.jit``.

A structural-only example: ``n_workers`` compute tiles produce 8-element
sub-tiles which are joined into a 24-element tile in the mem tile via
``of_out.prod().join(...)``, then forwarded to the shim.  Each worker
just acquires + releases its tile (the data path is the point — there
is no compute).  Useful for inspecting the generated MLIR for the join
pattern.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import ObjectFifo, Out, Program, Runtime, Worker
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

n_workers = 3

data_ty = np.ndarray[(48,), np.dtype[np.int32]]
tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]


@iron.jit
def join_L2(c_out: Out):
    of_offsets = [8 * w for w in range(n_workers)]

    of_out = ObjectFifo(tile24_ty, name="out")
    of_outs = of_out.prod().join(
        of_offsets,
        obj_types=[tile8_ty] * n_workers,
        names=[f"out{w}" for w in range(n_workers)],
    )

    def core_fn(of_out):
        elem = of_out.acquire(1)
        of_out.release(1)

    workers = [Worker(core_fn, [of_outs[w].prod()]) for w in range(n_workers)]

    rt = Runtime()
    with rt.sequence(data_ty) as c:
        rt.start(*workers)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _emit_mlir(opts):
    c_out = iron.zeros(48, dtype=np.int32, device="npu")
    print(join_L2.as_mlir(c_out))


def main():
    p = argparse.ArgumentParser(prog="join (L2 join) structural example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    if not getattr(opts, "emit_mlir", False):
        opts.emit_mlir = True
    run_design_cli(
        join_L2,
        opts,
        compile_kwargs={},
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
