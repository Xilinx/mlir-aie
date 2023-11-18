# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s
# REQUIRES: python_passes


import aie
from aie.ir import *
from aie.dialects.aie import *
from aie.passmanager import PassManager
from aie._mlir_libs import _aie_python_passes

from typing import List


def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        module = Module.create()
        print("\nTEST:", f.__name__)
        with InsertionPoint(module.body):
            f()
        print(module)


def testPythonPassDemo():
    # CHECK-LABEL: testPythonPassDemo
    print("\nTEST: testPythonPassDemo")

    def print_ops(op):
        print(op.name)

    module = """
    module {
      %12 = AIE.tile(1, 2)
      %buf = AIE.buffer(%12) : memref<256xi32>
      %4 = AIE.core(%12)  {
        %0 = arith.constant 0 : i32
        %1 = arith.constant 0 : index
        memref.store %0, %buf[%1] : memref<256xi32>
        AIE.end
      }
    }
    """

    # CHECK: AIE.tile
    # CHECK: AIE.buffer
    # CHECK: arith.constant
    # CHECK: arith.constant
    # CHECK: memref.store
    # CHECK: AIE.end
    # CHECK: AIE.core
    # CHECK: builtin.module
    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        _aie_python_passes.register_python_pass_demo_pass(print_ops)
        mlir_module = Module.parse(module)
        PassManager.parse("builtin.module(python-pass-demo)").run(mlir_module.operation)


testPythonPassDemo()
