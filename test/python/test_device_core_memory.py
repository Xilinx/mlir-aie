# test_device_core_memory.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Device.core_memory_banks and core_memory_bank_bytes: the banks the allocator places in."""

import pytest

from aie.ir import Context, Location, Module
from aie.iron.device import NPU1, NPU1Col1, NPU2, NPU2Col1
from aie.passmanager import PassManager


@pytest.mark.parametrize("device", [NPU1, NPU1Col1, NPU2, NPU2Col1])
def test_four_banks_of_16_kb(device):
    dev = device()
    assert dev.core_memory_banks == 4
    assert dev.core_memory_bank_bytes == 16 * 1024
    assert dev.core_memory_banks * dev.core_memory_bank_bytes == dev.core_memory_bytes


@pytest.mark.parametrize("device", [NPU1Col1, NPU2Col1])
def test_a_buffer_pinned_to_a_bank_lands_inside_it(device):
    dev = device()
    banks, bank_bytes = dev.core_memory_banks, dev.core_memory_bank_bytes
    buffers = "\n".join(
        f'%b{k} = aie.buffer(%t) {{sym_name = "b{k}", mem_bank = {k} : i32}}'
        f" : memref<{bank_bytes // 2}xi8>"
        for k in range(banks)
    )
    with Context(), Location.unknown():
        module = Module.parse(f"""
            aie.device({dev.name}) {{
              %t = aie.tile(0, {dev.core_rows[0]})
              {buffers}
              aie.core(%t) {{ aie.end }}
            }}""")
        PassManager.parse(
            "builtin.module(aie.device(aie-assign-buffer-addresses))"
        ).run(module.operation)
        placed = {
            op.attributes["sym_name"].value: op.attributes["address"].value
            for op in module.body.operations[0].regions[0].blocks[0].operations
            if op.operation.name == "aie.buffer"
        }
    assert sorted(placed) == [f"b{k}" for k in range(banks)]
    for k in range(banks):
        address = placed[f"b{k}"]
        assert k * bank_bytes <= address
        assert address + bank_bytes // 2 <= (k + 1) * bank_bytes
