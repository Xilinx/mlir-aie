# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s --compile --xchesscc --no-link --basic-alloc-scheme -nv | FileCheck %s

# CHECK: xchesscc_wrapper aie

import sys

import aie.compiler.aiecc.main as aiecc
from aie.ir import Context, Location, Module

module = """
module {
  %12 = aie.tile(1, 2)
  %buf = aie.buffer(%12) : memref<256xi32>
  %4 = aie.core(%12)  {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 0 : index
    memref.store %0, %buf[%1] : memref<256xi32>
    aie.end
  }
}
"""

with Context() as ctx, Location.unknown():
    mlir_module = Module.parse(module)

aiecc.run(mlir_module, sys.argv[1:])
