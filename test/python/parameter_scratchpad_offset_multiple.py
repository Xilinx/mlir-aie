# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s %t.params | FileCheck %s

# An offset parameter into bf16 elements, without a device: the BD address
# drops its low two bits, so an odd element count would read one element
# early. The lowering records that the value must be even and the host
# library refuses an odd one, while an offset into i32 elements takes any
# value in its range. A params file from before the multiples were written
# still loads.

import argparse

import numpy as np

from aie._mlir_libs._parameter_scratchpad import ParameterScratchpad
from aie.dialects import aiex  # noqa: F401  registers the AIE dialects
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager

SOURCE = """
aiex.scratchpad_parameter @row : i32
aiex.scratchpad_parameter @word : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%a : memref<1024xbf16>, %b : memref<1024xi32>) {
    %ta = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%a : memref<1024xbf16> offset = 0 len = 64) {offset_parameter = @row}
      aie.end
    } {issue_token = true}
    %tb = aiex.dma_configure_task(%t, MM2S, 1) {
      aie.dma_bd(%b : memref<1024xi32> offset = 0 len = 64) {offset_parameter = @word}
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%ta)
    aiex.dma_start_task(%tb)
    aiex.dma_await_task(%ta)
    aiex.dma_await_task(%tb)
  }
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params", help="where the lowering writes params.txt")
    args = parser.parse_args()

    with Context(), Location.unknown():
        module = Module.parse(SOURCE)
        PassManager.parse(
            "builtin.module(aie-lower-scratchpad-parameters"
            f"{{output-params-file={args.params}}},"
            "aie.device(aie-assign-runtime-sequence-bd-ids,"
            "aie-dma-tasks-to-npu,aie-dma-to-npu))"
        ).run(module.operation)

    # CHECK: row 0 i32 addr 0 960
    # CHECK: word 1 i32 addr 0 960
    # The multiples come last, after the one command.
    # CHECK: main:sequence 2 row word
    # CHECK-NEXT: 1
    # CHECK-NEXT: row 2
    with open(args.params) as f:
        text = f.read()
    print(text)

    table = np.zeros(2, dtype=np.uint32)
    params = ParameterScratchpad(table, args.params)

    # CHECK: row=6 word=0x6 word=7 word=0x7
    params.write_bytes("row", np.int32(6).tobytes())
    params.write_bytes("word", np.int32(7).tobytes())
    print(f"row=6 word={table[0]:#x} word=7 word={table[1]:#x}")

    # CHECK: refused 7: {{.*}}value 7 of 'row' is not a multiple of 2 elements
    # CHECK: kept word=0x6
    try:
        params.write_bytes("row", np.int32(7).tobytes())
    except ValueError as e:
        print(f"refused 7: {e}")
    print(f"kept word={table[0]:#x}")

    # CHECK: old file row=7 word=0x7
    old = args.params + ".old"
    with open(old, "w") as f:
        f.write("\n".join(text.splitlines()[:-2]) + "\n")
    params = ParameterScratchpad(table, old)
    params.write_bytes("row", np.int32(7).tobytes())
    print(f"old file row=7 word={table[0]:#x}")


main()
