# ext_to_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""External mem -> compute tile data movement — ``@iron.jit``.

A 48-element int32 vector is streamed from external memory into the
compute tile in 24-element tiles; each tile is incremented by 1 and
returned.  With input ``all-1s`` the host receives ``all-2s``.
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

tile_ty = np.ndarray[(24,), np.dtype[np.int32]]
data_ty = np.ndarray[(48,), np.dtype[np.int32]]


@iron.jit
def ext_to_core(a_in: In, c_out: Out):
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(24):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    my_worker = Worker(core_fn, [of_in.cons(), of_out.prod()])

    rt = Runtime()

    def sequence(a, c):
        of_in.prod().fill(a)
        of_out.cons().drain(c, wait=True)

    rt.sequence(sequence, [data_ty, data_ty])

    return Program(iron.get_current_device(), rt, workers=[my_worker]).resolve_program()


def _run_and_verify(opts):
    a_in = iron.ones(48, dtype=np.int32, device="npu")
    c_out = iron.zeros(48, dtype=np.int32, device="npu")
    ext_to_core(a_in, c_out)
    assert_pass(
        c_out.numpy(),
        np.full(48, 2, dtype=np.int32),
        fail_msg="ext_to_core output mismatch",
    )


def main():
    p = argparse.ArgumentParser(prog="external mem -> core example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        ext_to_core,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
