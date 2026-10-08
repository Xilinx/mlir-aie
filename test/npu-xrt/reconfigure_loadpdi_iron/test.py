# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: ryzen_ai_npu2, xrt_python_bindings, peano
# RUN: %run_on_npu2% %pytest %s

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import (
    CompileTime,
    DeviceConfiguration,
    InOut,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1

Chunk = np.ndarray[(4,), np.dtype[np.int32]]
Tensor = np.ndarray[(16,), np.dtype[np.int32]]


def add_configuration(name, value):
    input_fifo = ObjectFifo(Chunk, name=f"{name}_in")
    output_fifo = ObjectFifo(Chunk, name=f"{name}_out")

    def core(input_handle, output_handle):
        input_element = input_handle.acquire(1)
        output_element = output_handle.acquire(1)
        for index in range_(4):
            output_element[index] = input_element[index] + value
        input_handle.release(1)
        output_handle.release(1)

    worker = Worker(core, [input_fifo.cons(), output_fifo.prod()])

    def sequence(data, input_handle, output_handle):
        input_handle.fill(data)
        output_handle.drain(data, wait=True)

    runtime = Runtime(
        sequence,
        [Chunk, input_fifo.prod(), output_fifo.cons()],
        name=f"{name}_sequence",
    )
    configuration = DeviceConfiguration(
        name,
        NPU2Col1(),
        workers=[worker],
        runtimes=[runtime],
    )
    return configuration, runtime


@iron.jit(full_elf=True)
def reconfigure_add(data: InOut, *, reconfiguration_mode: CompileTime[str]):
    add_two, add_two_sequence = add_configuration("add_two", 2)
    add_three, add_three_sequence = add_configuration("add_three", 3)

    def coordinator(tensor):
        with add_two.configure():
            add_two_sequence.call(tensor.window(0, (4,)))
            add_two_sequence.call(tensor.window(12, (4,)))
        with add_three.configure():
            add_three_sequence.call(tensor.window(4, (4,)))
            add_three_sequence.call(tensor.window(12, (4,)))

    entry = Runtime(coordinator, [Tensor])
    main = DeviceConfiguration("main", NPU2Col1(), runtimes=[entry])
    return Program.compose(
        [main, add_two, add_three],
        entry=entry,
        reconfiguration_mode=reconfiguration_mode,
    ).resolve_program()


@pytest.mark.parametrize(
    "reconfiguration_mode",
    ["load-pdi", "expand-load-pdis", "control-packets"],
)
def test_reconfigure_add(reconfiguration_mode):
    data = iron.arange(16, dtype=np.int32, device="npu")

    reconfigure_add(data, reconfiguration_mode=reconfiguration_mode)
    data.to("cpu")

    expected = np.arange(16, dtype=np.int32)
    expected[0:4] += 2
    expected[4:8] += 3
    expected[12:16] += 5
    np.testing.assert_array_equal(data.numpy(), expected)
