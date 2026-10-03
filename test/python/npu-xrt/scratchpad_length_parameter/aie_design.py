# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: a transfer length set at runtime via length_parameter.
#
# One parameter @tiles sizes both the DMAs and the core's loop; @start offsets
# the input. With a static length of 0, the DMAs move exactly `tiles` tiles of
# 8 i32 values from value `start` on, and the core adds one to each value:
#
#   start = 0, tiles = 0 -> nothing
#   start = 5, tiles = 3 -> 24 values: 6..29
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.dialects.aiex import npu_load_pdi
from aie.dialects import arith
from aie.ir import IndexType

N = 256
TILE = 8


def design():
    device_name = "test"

    buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    start = ScratchpadParameter("start", np.int32)
    tiles = ScratchpadParameter("tiles", np.int32)

    of_in = ObjectFifo(tile_ty, name="objfifo_in")
    of_out = ObjectFifo(tile_ty, name="objfifo_out")

    def core_fn(of_in, of_out, tiles):
        count = arith.index_cast(IndexType.get(), tiles.read())
        for _ in range_(count):
            in_elem = of_in.acquire(1)
            out_elem = of_out.acquire(1)
            for i in range_(TILE):
                out_elem[i] = in_elem[i] + 1
            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_out.prod(), tiles],
        while_true=False,
    )

    def sequence(in_tensor, out_tensor, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=device_name)

        # The sizes give the shape of one tile; @tiles counts them.
        pattern = dict(
            offset=0,
            sizes=[1, 1, 1, TILE],
            strides=[0, 0, 0, 1],
            transfer_len=0,
            length_parameter=tiles,
            length_unit=TILE,
        )
        in_h.fill(in_tensor, offset_parameter=start, **pattern)
        out_h.drain(out_tensor, wait=True, **pattern)

    rt = Runtime(sequence, [buf_ty, buf_ty, of_in.prod(), of_out.cons()])

    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program(
        device_name=device_name
    )

    # Insert empty device to force PDI reload
    mlir_text = str(module)
    empty_device = "  aie.device(npu2) @empty { }\n"
    mlir_text = mlir_text.replace("module {\n", "module {\n" + empty_device, 1)
    return mlir_text


mlir_text = design()
print(mlir_text)
