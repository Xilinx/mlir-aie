# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test slicing an ExternalBuffer, and handing the result to a Bd.

A slice of a buffer is still a buffer: it can be copied from wherever the whole
one can, and the two share the one declaration rather than each emitting an
aie.external_buffer of their own.

Bd takes the access pattern directly, so a descriptor over part of a buffer
says the part rather than restating the offset, length, sizes and strides it
implies.
"""

import numpy as np
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.iron import (
    Bd,
    DmaChannel,
    ExternalBuffer,
    Program,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1, Tile

devmem_ty = np.ndarray[(16, 16, 512), np.dtype[np.int8]]


print("\nTEST: a_slice_is_a_buffer")
whole = ExternalBuffer(devmem_ty, address=0x8000_0000, name="devmem")
part = whole[0::2, 1::2, ...]
print(f"whole tap: {list(whole.tap.sizes)} offset {whole.tap.offset}")
print(f"part  tap: {list(part.tap.sizes)} offset {part.tap.offset}")

# CHECK-LABEL: a_slice_is_a_buffer
# CHECK: whole tap: [16, 16, 512] offset 0
# CHECK: part  tap: [8, 8, 512] offset 512

# Slicing a slice would measure against the whole buffer's shape rather than
# compose with the first, so it is refused instead of quietly answering.
try:
    part[0:2]
except ValueError as e:
    print(f"nested: {e}")

# CHECK: nested: devmem is already a slice


print("\nTEST: one_declaration_and_bd_takes_the_tap")
rt = Runtime(lambda: None, [])
shim = Tile(0, 0)
rt.add_external_buffer(whole)
rt.add_tile_dma(
    TileDma(
        tile=shim,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                loop=False,
                bds=[Bd(buffer=whole, tap=part.tap)],
            )
        ],
    )
)
print(Program(NPU2Col1(), rt).resolve_program())

# Both names refer to one declaration, so only one is emitted.
# CHECK-LABEL: one_declaration_and_bd_takes_the_tap
# CHECK:     aie.external_buffer {address = 2147483648 : i64, sym_name = "devmem"}
# CHECK-NOT: aie.external_buffer
# CHECK:     aie.dma_bd(%devmem
# CHECK-SAME: offset = 512 len = 32768 sizes = [8, 8, 512] strides = [16384, 1024, 1]


print("\nTEST: tap_and_explicit_geometry_are_exclusive")
try:
    Bd(buffer=whole, tap=part.tap, offset=4)
except ValueError as e:
    print(f"both: {e}")

# CHECK-LABEL: tap_and_explicit_geometry_are_exclusive
# CHECK: both: Bd.tap already says the offset, length, sizes and strides
