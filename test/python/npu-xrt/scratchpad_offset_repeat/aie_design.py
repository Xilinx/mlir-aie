# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: one shim DMA task per path with a runtime offset, a runtime
# repeat count and a runtime length together. Path A steps its outer dimension
# (stride ROW); path B repeats the same block (stride 0). Each is forwarded
# through a mem tile and drained with a runtime length.
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import DeviceConfiguration, ObjectFifo, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter

N = 1024
TILE = 16
ITERS = 8
ROW = 128


def design() -> str:
    buf_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    off = ScratchpadParameter("off", np.int32)
    n = ScratchpadParameter("n", np.int32)
    reps_a = ScratchpadParameter("reps_a", np.int32)
    reps_b = ScratchpadParameter("reps_b", np.int32)
    tiles_a = ScratchpadParameter("tiles_a", np.int32)
    tiles_b = ScratchpadParameter("tiles_b", np.int32)

    a_in = ObjectFifo(tile_ty, name="a_in")
    a_out = a_in.cons().forward(name="a_out")
    b_in = ObjectFifo(tile_ty, name="b_in")
    b_out = b_in.cons().forward(name="b_out")

    def sequence(src, dst_a, dst_b, a_in, b_in, a_out, b_out):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref="test")
        stepped = TensorAccessPattern((N,), 0, [ITERS, 1, 1, TILE], [ROW, 0, 0, 1])
        same = TensorAccessPattern((N,), 0, [ITERS, 1, 1, TILE], [0, 0, 0, 1])
        linear = TensorAccessPattern((N,), 0, [1, 1, 1, TILE], [0, 0, 0, 1])
        a_in.fill(
            src,
            tap=stepped,
            offset_parameter=off,
            length_parameter=n,
            repeat_parameter=reps_a,
        )
        b_in.fill(
            src,
            tap=same,
            offset_parameter=off,
            length_parameter=n,
            repeat_parameter=reps_b,
        )
        a_out.drain(dst_a, tap=linear, wait=True, length_parameter=tiles_a)
        b_out.drain(dst_b, tap=linear, wait=True, length_parameter=tiles_b)

    rt = Runtime(
        sequence,
        [
            buf_ty,
            buf_ty,
            buf_ty,
            a_in.prod(),
            b_in.prod(),
            a_out.cons(),
            b_out.cons(),
        ],
        name="sequence",
        implicit_configure=False,
    )
    config = DeviceConfiguration("test", NPU2Col1(), runtimes=[rt])
    empty = DeviceConfiguration("empty", NPU2Col1())
    return str(Program.compose([empty, config], entry=rt).resolve_program())


if __name__ == "__main__":
    print(design())
