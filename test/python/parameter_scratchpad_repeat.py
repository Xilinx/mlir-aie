# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s %t.params | FileCheck %s

# A repeat parameter end to end, without a device: the lowered instruction
# stream updates the MM2S task queue from the parameter's state table entry
# (bd 4, since the S2MM task holds bd 0), and the host library writes the
# parameter shifted into the queue word's repeat field, so the update adds
# exactly the push's repeat count. It refuses a count past the task's own,
# which the lowering records as the parameter's range.

import argparse

import numpy as np

from aie._mlir_libs._parameter_scratchpad import ParameterScratchpad
from aie.dialects.aie import translate_npu_to_binary
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager
from aie.utils.txn_trace import trace

SOURCE = """
aiex.scratchpad_parameter @tiles : i32
aiex.scratchpad_parameter @reps : i32
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0 : memref<1024xi32>, %arg1 : memref<1024xi32>) {
    %out = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 0) {length_parameter = @tiles, length_unit = 64 : i32}
      aie.end
    } {issue_token = true}
    %in = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64) {iteration = #aie.bd_iteration<size = 16, stride = 64, current = 0>}
      aie.end
    } {repeat_count = 15 : i32}
    aiex.dma_start_task(%out)
    aiex.dma_start_task(%in) {repeat_parameter = @reps}
    aiex.dma_await_task(%out)
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
        words = translate_npu_to_binary(module.operation)

    # CHECK: update_reg (0, 0, 64, 118784)
    # CHECK: push (0,0) S2MM ch0 token
    # CHECK: update_reg (1, 1, 4, 119316)
    # CHECK: wait (0,0) S2MM ch0
    for event in trace(words):
        if event.kind != "write":
            print(event)

    # CHECK: reps 1 i32 repeat 0 15
    with open(args.params) as f:
        print(f.read())

    table = np.zeros(2, dtype=np.uint32)
    params = ParameterScratchpad(table, args.params)

    # CHECK: reps=6 word=0x60000 tiles=7 word=0x7
    params.write_bytes("reps", np.int32(6).tobytes())
    params.write_bytes("tiles", np.int32(7).tobytes())
    print(f"reps=6 word={table[1]:#x} tiles=7 word={table[0]:#x}")

    # CHECK: reps=15 word=0xf0000
    params.write_bytes("reps", np.int32(15).tobytes())
    print(f"reps=15 word={table[1]:#x}")

    # CHECK: refused 16: {{.*}}outside [0, 15], its task's largest repeat count
    # CHECK: refused -1: {{.*}}outside [0, 15], its task's largest repeat count
    # CHECK: kept word=0xf0000
    for reps in (16, -1):
        try:
            params.write_bytes("reps", np.int32(reps).tobytes())
        except ValueError as e:
            print(f"refused {reps}: {e}")
    print(f"kept word={table[1]:#x}")


main()
