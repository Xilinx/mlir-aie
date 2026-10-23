# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design: read TILE words at a runtime offset from a 2.125 GiB int32 host
# buffer (the size of a 128k-token Llama 3.2 1B KV cache), forwarded through a
# mem tile to a small output.
#
# Usage:
#   python3 aie_design.py > aie.mlir

import numpy as np

from aie.dialects.aiex import npu_load_pdi
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import DeviceConfiguration, ObjectFifo, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter

WORDS = 2_281_701_376 // 4
TILE = 16


def design() -> str:
    big_ty = np.ndarray[(WORDS,), np.dtype[np.int32]]
    out_ty = np.ndarray[(TILE,), np.dtype[np.int32]]

    off = ScratchpadParameter("off", np.int32)
    of_in = ObjectFifo(out_ty, name="objfifo_in")
    of_out = of_in.cons().forward(name="objfifo_out")

    def sequence(big, out, in_h, out_h):
        npu_load_pdi(device_ref="empty")
        npu_load_pdi(device_ref="test")
        block = TensorAccessPattern((WORDS,), 0, [1, 1, 1, TILE], [0, 0, 0, 1])
        in_h.fill(big, tap=block, offset_parameter=off)
        out_h.drain(out, wait=True)

    rt = Runtime(
        sequence,
        [big_ty, out_ty, of_in.prod(), of_out.cons()],
        name="sequence",
        implicit_configure=False,
    )
    config = DeviceConfiguration("test", NPU2Col1(), runtimes=[rt])
    empty = DeviceConfiguration("empty", NPU2Col1())
    return str(Program.compose([empty, config], entry=rt).resolve_program())


if __name__ == "__main__":
    print(design())
