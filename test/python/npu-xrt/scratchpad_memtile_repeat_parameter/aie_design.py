# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# A mem tile's S2MM and MM2S tasks each run @reps + 1 of up to 16 tiles of
# 64 values, and the drain moves @tiles tiles back: reps = 6, tiles = 7 gives
# 0..447. A mem tile task that ran its configured 16 runs would wait for data
# the fill never sends, and the call would not finish.
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.dialects.aie import BdIteration
from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Bd, Buffer, DeviceConfiguration, Flow, Program, Runtime
from aie.iron.device import NPU2Col1, Tile
from aie.iron.scratchpad_parameter import ScratchpadParameter

N = 1024
TILE = 64


def design():
    device_name = "test"

    buf_ty = np.ndarray[(N,), np.dtype[np.int32]]

    reps = ScratchpadParameter("reps", np.int32)
    tiles = ScratchpadParameter("tiles", np.int32)

    shim, mem = Tile(0, 0), Tile(0, 1)
    held = Buffer(buf_ty, tile=mem, name="held")
    into = Flow(shim, mem)
    out = Flow(mem, shim)

    def sequence(in_tensor, out_tensor):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref=device_name)

        stepped = TensorAccessPattern((N,), 0, [N // TILE, 1, 1, TILE], [TILE, 0, 0, 1])
        tile = TensorAccessPattern((N,), 0, [1, 1, 1, TILE], [0, 0, 0, 1])
        each = TensorAccessPattern((N,), 0, [TILE], [1])
        step = BdIteration(N // TILE, TILE, 0)

        land = into.endpoint(mem).task(
            Bd(held, tap=each, iteration=step), runs=N // TILE, wait=True
        )
        land.start(repeat_parameter=reps)
        into.fill(in_tensor, tap=stepped, repeat_parameter=reps)
        land.await_()
        send = out.endpoint(mem).task(
            Bd(held, tap=each, iteration=step), runs=N // TILE
        )
        send.start(repeat_parameter=reps)
        out.drain(out_tensor, tap=tile, wait=True, length_parameter=tiles)

    rt = Runtime(
        sequence,
        [buf_ty, buf_ty],
        name="sequence",
        implicit_configure=False,
    )
    rt.add_flow(into)
    rt.add_flow(out)

    config = DeviceConfiguration(device_name, NPU2Col1(), runtimes=[rt])
    # The firmware skips reloading the PDI it loaded last, so the sequence
    # loads an empty device first.
    empty = DeviceConfiguration("empty", NPU2Col1())
    return str(Program.compose([empty, config], entry=rt).resolve_program())


mlir_text = design()
print(mlir_text)
