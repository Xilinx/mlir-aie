# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: ryzen_ai_npu2, xrt_python_bindings, peano
# RUN: %run_on_npu2% %pytest %s

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Bd,
    Buffer,
    CompileTime,
    DeviceConfiguration,
    DmaChannel,
    Flow,
    Out,
    Program,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1, Tile

Chunk = np.ndarray[(4,), np.dtype[np.int32]]
Tensor = np.ndarray[(8,), np.dtype[np.int32]]
VALUES = np.array([11, 22, 33, 44], dtype=np.int32)


def memtile_configuration(name, initial_value=None):
    shim = Tile(0, 0, tile_type=AIETileType.ShimNOCTile)
    memtile = Tile(0, 1, tile_type=AIETileType.MemTile)
    resident = Buffer(
        Chunk,
        initial_value=initial_value,
        name=f"{name}_resident",
        tile=memtile,
        address=0,
    )
    output = Flow(memtile, shim, src_channel=0, dst_channel=0)
    memtile_dma = TileDma(
        memtile,
        [DmaChannel(DMAChannelDir.MM2S, output.endpoint(memtile), [Bd(resident)])],
    )

    def sequence(data):
        output.drain(data, wait=True)

    runtime = Runtime(sequence, [Chunk], name="sequence")
    runtime.add_flow(output)
    runtime.add_tile_dma(memtile_dma)
    configuration = DeviceConfiguration(name, NPU2Col1(), runtimes=[runtime])
    return configuration, runtime


@iron.jit(full_elf=True)
def persistent_memtile(data: Out, *, expand_load_pdis: CompileTime[bool | None]):
    initialized, initialized_sequence = memtile_configuration(
        "yield_const_memtile", VALUES
    )
    uninitialized, uninitialized_sequence = memtile_configuration(
        "yield_uninitialized_memtile"
    )

    def coordinator(tensor):
        with initialized.configure():
            initialized_sequence.call(tensor.window(0, (4,)))
        with uninitialized.configure():
            uninitialized_sequence.call(tensor.window(4, (4,)))

    entry = Runtime(coordinator, [Tensor])
    main = DeviceConfiguration("main", NPU2Col1(), runtimes=[entry])
    return Program.compose(
        [main, initialized, uninitialized],
        entry=entry,
        expand_load_pdis=expand_load_pdis,
    ).resolve_program()


@pytest.mark.parametrize("expand_load_pdis", [None, False, True])
def test_persistent_memtile(expand_load_pdis):
    data = iron.zeros(8, dtype=np.int32, device="npu")

    persistent_memtile(data, expand_load_pdis=expand_load_pdis)
    data.to("cpu")

    np.testing.assert_array_equal(data.numpy(), np.tile(VALUES, 2))
