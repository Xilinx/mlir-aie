# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: ryzen_ai_npu2, xrt_python_bindings, peano
# RUN: %run_on_npu2% %pytest %s

# The reconfigure_loadpdi_iron program with teardown and an init entry. The host
# runtime runs main:reconfig_init on each hardware context it creates. Under
# control packet reconfiguration, main:reconfig_init installs the control
# overlay.

import aie.iron as iron
import numpy as np
import pyxrt
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
from aie.utils import DefaultNPURuntime

Chunk = np.ndarray[(4,), np.dtype[np.int32]]
Tensor = np.ndarray[(8,), np.dtype[np.int32]]


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


@iron.jit
def reconfigure_add(data: InOut, *, ctrl_pkt: CompileTime[bool]):
    add_two, add_two_sequence = add_configuration("add_two", 2)
    add_three, add_three_sequence = add_configuration("add_three", 3)

    def coordinator(tensor):
        with add_two.configure():
            add_two_sequence.call(tensor.window(0, (4,)))
        with add_three.configure():
            add_three_sequence.call(tensor.window(4, (4,)))

    entry = Runtime(coordinator, [Tensor])
    main = DeviceConfiguration("main", NPU2Col1(), runtimes=[entry])
    return Program.compose(
        [main, add_two, add_three],
        entry=entry,
        expand_load_pdis=None if ctrl_pkt else True,
        load_pdi_to_ctrl_pkt=True if ctrl_pkt else None,
        reconfig_teardown=True,
        reconfig_init_entry=True,
    ).resolve_program()


def expected():
    values = np.arange(8, dtype=np.int32)
    values[0:4] += 2
    values[4:8] += 3
    return values


@pytest.fixture
def looked_up(monkeypatch):
    """The names the host runtime looks up with pyxrt.ext.kernel.

    The runtime starts with no hardware context, so a rerun loads afresh.
    """
    DefaultNPURuntime.cleanup()
    names = []
    lookup = pyxrt.ext.kernel

    def record(context, name):
        names.append(name)
        return lookup(context, name)

    monkeypatch.setattr(pyxrt.ext, "kernel", record)
    return names


@pytest.mark.parametrize("ctrl_pkt", [False, True])
def test_init_entry_runs_on_each_new_context(looked_up, ctrl_pkt):
    for _ in range(2):
        data = iron.arange(8, dtype=np.int32, device="npu")
        reconfigure_add(data, ctrl_pkt=ctrl_pkt)
        data.to("cpu")
        np.testing.assert_array_equal(data.numpy(), expected())
    assert looked_up == ["main:sequence", "main:reconfig_init"]

    # A new context, as after an IOCTL EINVAL, runs main:reconfig_init again.
    DefaultNPURuntime.cleanup()
    data = iron.arange(8, dtype=np.int32, device="npu")
    reconfigure_add(data, ctrl_pkt=ctrl_pkt)
    data.to("cpu")
    np.testing.assert_array_equal(data.numpy(), expected())
    assert looked_up == ["main:sequence", "main:reconfig_init"] * 2
