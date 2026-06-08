# distribute_and_join_L2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""``cons().split(...)`` + ``prod().join(...)`` round trip — ``@iron.jit``.

A 48-element int32 vector enters at the shim as 24-element tiles, lands
in the mem tile, fans out 8-element sub-tiles to 3 compute tiles via
``cons().split(...)``, each tile adds 1, and the per-tile outputs are
joined back to 24-element tiles via ``prod().join(...)`` for return to
the shim.  With input ``all-1s`` the host receives ``all-2s``.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

n_workers = 3

data_ty = np.ndarray[(48,), np.dtype[np.int32]]
tile24_ty = np.ndarray[(24,), np.dtype[np.int32]]
tile8_ty = np.ndarray[(8,), np.dtype[np.int32]]


@iron.jit
def distribute_and_join_L2(a_in: In, c_out: Out):
    of_offsets = [8 * w for w in range(n_workers)]

    of_in = ObjectFifo(tile24_ty, name="in")
    of_ins = of_in.cons().split(
        of_offsets,
        obj_types=[tile8_ty] * n_workers,
        names=[f"in{w}" for w in range(n_workers)],
    )

    of_out = ObjectFifo(tile24_ty, name="out")
    of_outs = of_out.prod().join(
        of_offsets,
        obj_types=[tile8_ty] * n_workers,
        names=[f"out{w}" for w in range(n_workers)],
    )

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(8):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[w].cons(), of_outs[w].prod()]) for w in range(n_workers)
    ]

    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a, c):
        rt.start(*workers)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _run_and_verify(opts):
    a_in = iron.ones(48, dtype=np.int32, device="npu")
    c_out = iron.zeros(48, dtype=np.int32, device="npu")
    distribute_and_join_L2(a_in, c_out)
    assert_pass(
        c_out.numpy(),
        np.full(48, 2, dtype=np.int32),
        fail_msg="distribute_and_join_L2 output mismatch",
    )


def _emit_mlir(opts):
    a_in = iron.zeros(48, dtype=np.int32, device="npu")
    c_out = iron.zeros(48, dtype=np.int32, device="npu")
    print(distribute_and_join_L2.as_mlir(a_in, c_out))


def main():
    p = argparse.ArgumentParser(prog="distribute+join (L2 split+join) example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        distribute_and_join_L2,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
