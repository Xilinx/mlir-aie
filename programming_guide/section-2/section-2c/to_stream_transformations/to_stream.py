# to_stream.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""``dims_to_stream`` example — Iron API design with ``@iron.jit``.

Mirror of the ``dims_from_stream`` example: the core copies a 24-element
int32 vector through unchanged, and the core->memtile ObjectFifo's
``dims_to_stream=[(8, 1), (3, 8)]`` reshapes the output as it streams
back to the shim, producing the same (3, 8) -> (8, 3) transpose of the
input ``arange(24)``.
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

data_ty = np.ndarray[(24,), np.dtype[np.int32]]


@iron.jit
def to_stream(a_in: In, c_out: Out):
    of_in0 = ObjectFifo(data_ty, name="in0")
    of_in1 = of_in0.cons().forward(name="in1", obj_type=data_ty)

    of_out1 = ObjectFifo(data_ty, name="out1", dims_to_stream=[(8, 1), (3, 8)])
    of_out0 = of_out1.cons().forward(name="out0", obj_type=data_ty)

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(24):
            elem_out[i] = elem_in[i]
        of_in.release(1)
        of_out.release(1)

    my_worker = Worker(core_fn, [of_in1.cons(), of_out1.prod()])

    rt = Runtime()
    with rt.sequence(data_ty, data_ty) as (a, c):
        rt.start(my_worker)
        rt.fill(of_in0.prod(), a)
        rt.drain(of_out0.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _expected_output():
    return np.arange(24, dtype=np.int32).reshape(3, 8).T.reshape(-1)


def _run_and_verify(opts):
    a_in = iron.arange(24, dtype=np.int32, device="npu")
    c_out = iron.zeros(24, dtype=np.int32, device="npu")
    to_stream(a_in, c_out)
    assert_pass(c_out.numpy(), _expected_output(), fail_msg="to_stream mismatch")


def _emit_mlir(opts):
    a_in = iron.zeros(24, dtype=np.int32, device="npu")
    c_out = iron.zeros(24, dtype=np.int32, device="npu")
    print(to_stream.as_mlir(a_in, c_out))


def main():
    p = argparse.ArgumentParser(prog="dims_to_stream example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        to_stream,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
