# single_buffer.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Single-buffer (depth=1) ObjectFifo example — ``@iron.jit``.

Two workers pipelined through a single-buffer ObjectFifo: worker 1
writes 16 elements of value ``1`` into ``of_in``; worker 2 copies them
into ``of_out``.  Both FIFOs have ``depth=1`` (single buffer, no
ping-pong) — the section illustrates the depth knob.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

data_ty = np.ndarray[(16,), np.dtype[np.int32]]


@iron.jit
def single_buffer(c_out: Out):
    of_in = ObjectFifo(data_ty, name="in", depth=1)
    of_out = ObjectFifo(data_ty, name="out", depth=1)

    def producer(of_in):
        elem = of_in.acquire(1)
        for i in range_(16):
            elem[i] = 1
        of_in.release(1)

    def copier(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(16):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    w1 = Worker(producer, [of_in.prod()])
    w2 = Worker(copier, [of_in.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(data_ty) as c:
        rt.start(w1)
        rt.start(w2)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _run_and_verify(opts):
    c_out = iron.zeros(16, dtype=np.int32, device="npu")
    single_buffer(c_out)
    assert_pass(
        c_out.numpy(),
        np.ones(16, dtype=np.int32),
        fail_msg="single_buffer output mismatch",
    )


def main():
    p = argparse.ArgumentParser(prog="single_buffer (depth=1) example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        single_buffer,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
