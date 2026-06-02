# section-2/section-2e/aie2_multi.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Section-2e multi-core scale-up — ``@iron.jit``.

Same input+1 task as ``aie2.py``, but distributed across ``n_workers``
compute tiles.  The mem tile fans the 48-element input out to per-worker
16-element tiles via ``cons().split(...)`` and joins the per-worker
outputs back via ``prod().join(...)``.  Used by Section 2e to motivate
the single-core-to-multi-core scale-up.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

n_workers = 3
data_size = 48
tile_size = data_size // n_workers

data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]


@iron.jit
def section_2e_multi(a_in: In, b_out: Out):
    of_offsets = [tile_size * w for w in range(n_workers)]

    of_in = ObjectFifo(data_ty, name="in")
    of_ins = of_in.cons().split(
        of_offsets,
        obj_types=[tile_ty] * n_workers,
        names=[f"in{w}" for w in range(n_workers)],
    )

    of_out = ObjectFifo(data_ty, name="out")
    of_outs = of_out.prod().join(
        of_offsets,
        obj_types=[tile_ty] * n_workers,
        names=[f"out{w}" for w in range(n_workers)],
    )

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(tile_size):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[w].cons(), of_outs[w].prod()])
        for w in range(n_workers)
    ]

    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a, b):
        rt.start(*workers)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _run_and_verify(opts):
    a_in = iron.arange(data_size, dtype=np.int32, device="npu")
    b_out = iron.zeros(data_size, dtype=np.int32, device="npu")
    section_2e_multi(a_in, b_out)
    expected = np.arange(data_size, dtype=np.int32) + 1
    assert_pass(b_out.numpy(), expected, fail_msg="section_2e_multi output mismatch")


def _emit_mlir(opts):
    a_in = iron.zeros(data_size, dtype=np.int32, device="npu")
    b_out = iron.zeros(data_size, dtype=np.int32, device="npu")
    print(section_2e_multi.as_mlir(a_in, b_out))


def main():
    p = argparse.ArgumentParser(prog="Section 2e multi-core example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        section_2e_multi, opts, compile_kwargs={},
        run_and_verify=_run_and_verify, emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
