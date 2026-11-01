# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: a transfer's repeat count set at runtime via repeat_parameter.
#
# The fill steps through up to 16 tiles of 64 values (its tap's outer
# dimension) and runs @reps + 1 of them; a memtile forwards them, and the
# drain moves @tiles tiles back (length_parameter):
#
#   reps = 0, tiles = 1  -> 64 values: 0..63
#   reps = 6, tiles = 7  -> 448 values: 0..447
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.iron import DeviceConfiguration, ObjectFifo, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.helpers.taplib import TensorAccessPattern
from aie.dialects.aiex import npu_load_pdi

N = 1024
TILE = 64


def design():
    device_name = "test"

    buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    reps = ScratchpadParameter("reps", np.int32)
    tiles = ScratchpadParameter("tiles", np.int32)

    of_in = ObjectFifo(tile_ty, name="objfifo_in")
    of_out = of_in.cons().forward(name="objfifo_out")

    def sequence(in_tensor, out_tensor, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=device_name)

        stepped = TensorAccessPattern((N,), 0, [N // TILE, 1, 1, TILE], [TILE, 0, 0, 1])
        tile = TensorAccessPattern((N,), 0, [1, 1, 1, TILE], [0, 0, 0, 1])
        in_h.fill(in_tensor, tap=stepped, repeat_parameter=reps)
        out_h.drain(out_tensor, tap=tile, wait=True, length_parameter=tiles)

    rt = Runtime(
        sequence,
        [buf_ty, buf_ty, of_in.prod(), of_out.cons()],
        name="sequence",
        implicit_configure=False,
    )

    config = DeviceConfiguration(device_name, NPU2Col1(), runtimes=[rt])
    # The firmware skips reloading the PDI it loaded last, so the sequence
    # loads an empty device first.
    empty = DeviceConfiguration("empty", NPU2Col1())
    return str(Program.compose([empty, config], entry=rt).resolve_program())


mlir_text = design()
print(mlir_text)
