# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: xrt_python_bindings
# RUN: %python %s %t.params | FileCheck %s

# Two commands, each a runtime sequence of its own with its own scratchpad:
# @a, which only `first` reads, and @b, which only `second` reads, share slot
# 0. Each command's scratchpad holds its own parameters, and refuses another
# command's, whose slot it may share.

import argparse

import numpy as np

from aie._mlir_libs._parameter_scratchpad import ParameterScratchpad
from aie.dialects import aie, aiex  # noqa: F401 (registers what SOURCE uses)
from aie.ir import Context, Location, Module
from aie.passmanager import PassManager

SOURCE = """
aiex.scratchpad_parameter @a : i32
aiex.scratchpad_parameter @b : i32
aiex.scratchpad_parameter @shared : i32
aie.device(npu2) @design_a {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in_a (%t, MM2S, 0)
  aie.runtime_sequence @seq_a(%in: memref<1024xi32>) {
    aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
      {id = 0 : i64, metadata = @in_a, offset_parameter = @a} : memref<1024xi32>
    aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
      {id = 1 : i64, metadata = @in_a, offset_parameter = @shared} : memref<1024xi32>
  }
}
aie.device(npu2) @design_b {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @in_b (%t, MM2S, 0)
  aie.runtime_sequence @seq_b(%in: memref<1024xi32>) {
    aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
      {id = 0 : i64, metadata = @in_b, offset_parameter = @b} : memref<1024xi32>
    aiex.npu.dma_memcpy_nd(%in[0,0,0,0][1,1,1,64][0,0,0,1])
      {id = 1 : i64, metadata = @in_b, offset_parameter = @shared} : memref<1024xi32>
  }
}
aie.device(npu2) @main {
  aie.runtime_sequence @first(%in: memref<1024xi32>) {
    aiex.configure @design_a {
      aiex.run @seq_a(%in) : (memref<1024xi32>)
    }
  }
  aie.runtime_sequence @second(%in: memref<1024xi32>) {
    aiex.configure @design_b {
      aiex.run @seq_b(%in) : (memref<1024xi32>)
    }
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
            f"{{output-params-file={args.params}}})"
        ).run(module.operation)

    # CHECK: first ['a', 'shared'] a=7 at 0 shared=9 at 1
    # CHECK: second ['b', 'shared'] b=8 at 0 shared=9 at 1
    for command, name, value in (("first", "a", 7), ("second", "b", 8)):
        table = np.zeros(2, dtype=np.uint32)
        params = ParameterScratchpad(table, args.params, f"main:{command}")
        params.write_bytes(name, np.int32(value).tobytes())
        params.write_bytes("shared", np.int32(9).tobytes())
        print(
            command,
            sorted(params.names()),
            f"{name}={value} at {list(table).index(value)}",
            f"shared=9 at {list(table).index(9)}",
        )

    # CHECK: refused: ParameterScratchpad: command 'main:first' does not read parameter 'b'
    table = np.zeros(2, dtype=np.uint32)
    params = ParameterScratchpad(table, args.params, "main:first")
    try:
        params.write_bytes("b", np.int32(8).tobytes())
    except RuntimeError as error:
        print("refused:", error)


if __name__ == "__main__":
    main()
