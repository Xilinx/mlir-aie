# section-2/section-2e/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Section-2e single-core design — ``@iron.jit``.

One Worker takes a 48-element int32 vector and adds 1 to each element.
Data flows shim -> mem tile -> core -> mem tile -> shim via paired
``forward()`` ObjectFifos.  Used by Section 2e to motivate the
single-core-to-multi-core scale-up in ``aie2_multi.py``.
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

data_size = 48
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]


@iron.jit
def section_2e(a_in: In, b_out: Out):
    of_in = ObjectFifo(data_ty, name="in")
    of_in1 = of_in.cons().forward(obj_type=data_ty, name="in1")

    of_out1 = ObjectFifo(data_ty, name="out1")
    of_out = of_out1.cons().forward(obj_type=data_ty, name="out")

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(data_size):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    my_worker = Worker(core_fn, [of_in1.cons(), of_out1.prod()])

    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a, b):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _run_and_verify(opts):
    a_in = iron.arange(data_size, dtype=np.int32, device="npu")
    b_out = iron.zeros(data_size, dtype=np.int32, device="npu")
    section_2e(a_in, b_out)
    expected = np.arange(data_size, dtype=np.int32) + 1
    assert_pass(b_out.numpy(), expected, fail_msg="section_2e output mismatch")


def _emit_mlir(opts):
    a_in = iron.zeros(data_size, dtype=np.int32, device="npu")
    b_out = iron.zeros(data_size, dtype=np.int32, device="npu")
    print(section_2e.as_mlir(a_in, b_out))


def main():
    p = argparse.ArgumentParser(prog="Section 2e single-core example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        section_2e,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
