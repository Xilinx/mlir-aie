# test_prebuilt_device_configuration.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of a device built elsewhere, imported by DeviceConfiguration.from_module.

The imported design adds 7 in a device named ``made_elsewhere``; it is
composed under a new name with a design built here, which doubles. Either a
coordinator configures and runs the imported device and then the built one on
its output, or the imported runtime is the program's entry, which must load
its own device's PDI.
"""

import aie.iron as iron
import numpy as np
from aie.ir import Context, Module
from aie.iron import (
    CompileTime,
    DeviceConfiguration,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1

N = 1024
TILE = 16


def elementwise(op, name):
    """A Worker applying ``op`` to each element, and the Runtime that feeds it."""
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core(cin, cout):
        for _ in range_(N // TILE):
            x = cin.acquire(1)
            y = cout.acquire(1)
            for i in range_(TILE):
                y[i] = op(x[i])
            cin.release(1)
            cout.release(1)

    def seq(a, b, hin, hout):
        hin.fill(a)
        hout.drain(b, wait=True)

    worker = Worker(core, [of_in.cons(), of_out.prod()])
    rt = Runtime(
        seq,
        [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()],
        name=name,
        implicit_configure=False,
    )
    return worker, rt


def made_elsewhere() -> str:
    worker, rt = elementwise(lambda v: v + 7, "run")
    with Context():
        return str(
            Program(NPU2Col1(), rt, workers=[worker]).resolve_program(
                device_name="made_elsewhere"
            )
        )


def compose(text: str, entry_is_imported: bool):
    """The imported device beside a built one that doubles, with its entry."""
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    imported = DeviceConfiguration.from_module("imported", Module.parse(text))
    (run,) = imported.runtimes
    worker, double = elementwise(lambda v: v * 2, "double")

    def coordinate(A, B, C):
        with imported.configure():
            run.call(A, B)
        with built.configure():
            double.call(B, C)

    coordinator = Runtime(
        coordinate, [tensor_ty] * 3, name="coordinate", implicit_configure=False
    )
    built = DeviceConfiguration(
        "built", NPU2Col1(), workers=[worker], runtimes=[double, coordinator]
    )
    entry = run if entry_is_imported else coordinator
    return Program.compose([built, imported], entry=entry).resolve_program()


@iron.jit
def coordinated(a: In, b: Out, c: Out, *, text: CompileTime[str]):
    return compose(text, entry_is_imported=False)


@iron.jit
def imported_entry(a: In, b: Out, *, text: CompileTime[str]):
    return compose(text, entry_is_imported=True)


def test_a_coordinator_runs_an_imported_device_then_a_built_one():
    host = np.random.default_rng(2).integers(-(2**20), 2**20, N, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    b = iron.zeros(N, dtype=np.int32, device="npu")
    c = iron.zeros(N, dtype=np.int32, device="npu")
    coordinated(a, b, c, text=made_elsewhere())
    b.to("cpu")
    c.to("cpu")
    np.testing.assert_array_equal(b.numpy(), host + 7)
    np.testing.assert_array_equal(c.numpy(), (host + 7) * 2)


def test_an_imported_runtime_runs_as_the_entry():
    host = np.random.default_rng(3).integers(-(2**20), 2**20, N, dtype=np.int32)
    a = iron.tensor(host, dtype=np.int32, device="npu")
    b = iron.zeros(N, dtype=np.int32, device="npu")
    imported_entry(a, b, text=made_elsewhere())
    b.to("cpu")
    np.testing.assert_array_equal(b.numpy(), host + 7)
