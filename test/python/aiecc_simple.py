# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# REQUIRES: chess

# RUN: %PYTHON %s --compile --xchesscc --no-link -nv | FileCheck %s

# CHECK: xchesscc_wrapper aie

import sys

import aie.compiler.aiecc.main as aiecc
from aie.ir import Context, Location, Module

# Import dialects to ensure they are registered before parsing
import aie.dialects.aie  # noqa: F401
import aie.dialects.aiex  # noqa: F401

module = """
module {
  aie.device(xcvc1902) {
    %12 = aie.tile(1, 2)
    %buf = aie.buffer(%12) : memref<256xi32>
    %4 = aie.core(%12)  {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : index
      memref.store %0, %buf[%1] : memref<256xi32>
      aie.end
    }
  }
}
"""

with Context() as ctx, Location.unknown():
    mlir_module = Module.parse(module)

output = aiecc.run(mlir_module, sys.argv[1:])
if output:
    print(output)
