# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: the extent of a shim transfer set per run via size_parameters.
#
# The input is 256 i32 values [0, 1, ..., 255] and the transfers move 16-value
# tiles, at most 16 of them. Each run the host sets:
#
#   @n   the tiles the input transfer moves, on its D2 (dimension 1); only a
#        DMA uses it (kind addr)
#   @m   the pairs of tiles the core copies; the output transfer moves m tiles
#        into each half of the output, patching dimension 2 of a pattern that
#        iterates over the halves, so the patched length is per iteration. The
#        core reads it too (kind core), and the host writes n = 2 * m
#   @off the element offset the input transfer starts at, on the same BD
#
# so output[:16 * m] = input[off : off + 16 * m], output[128 : 128 + 16 * m]
# = the next 16 * m inputs, and the rest of the output is left as the host
# set it.
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker, sync_parameters
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern

N = 256
TILE = 16
MAX_TILES = 16


def design():
    device_name = "test"

    buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    n = ScratchpadParameter("n", np.int32)
    m = ScratchpadParameter("m", np.int32)
    off = ScratchpadParameter("off", np.int32)

    of_in = ObjectFifo(tile_ty, name="objfifo_in")
    of_out = ObjectFifo(tile_ty, name="objfifo_out")

    def core_fn(of_in, of_out, m):
        for _ in range_(m.read()):
            for _ in range(2):
                in_elem = of_in.acquire(1)
                out_elem = of_out.acquire(1)
                for i in range(TILE):
                    out_elem[i] = in_elem[i]
                of_in.release(1)
                of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), m], while_true=False)

    # The pattern of the most tiles a run may move; the parameter says how
    # many of its D2 steps a run takes.
    tiles = TensorAccessPattern(
        (N,), offset=0, sizes=[1, MAX_TILES, 1, TILE], strides=[0, TILE, 0, 1]
    )
    # Two halves of the output, the tiles of each on dimension 2: the build
    # lays it out as [2, MAX_TILES // 2, 1, TILE], the halves the iteration.
    halves = TensorAccessPattern(
        (N,),
        offset=0,
        sizes=[1, 2, MAX_TILES // 2, TILE],
        strides=[0, N // 2, TILE, 1],
    )

    def sequence(in_tensor, out_tensor, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=device_name)
        sync_parameters()
        in_h.fill(in_tensor, tap=tiles, offset_parameter=off, size_parameters={1: n})
        out_h.drain(out_tensor, tap=halves, size_parameters={2: m}, wait=True)

    rt = Runtime(sequence, [buf_ty, buf_ty, of_in.prod(), of_out.cons()])

    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program(
        device_name=device_name
    )

    # Insert an empty device to force a PDI reload, so the core runs again.
    mlir_text = str(module)
    empty_device = "  aie.device(npu2) @empty { }\n"
    mlir_text = mlir_text.replace("module {\n", "module {\n" + empty_device, 1)
    return mlir_text


mlir_text = design()
print(mlir_text)
