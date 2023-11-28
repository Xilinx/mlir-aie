# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

from aie.dialects.aiex import *


def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        module = Module.create()
        print("\nTEST:", f.__name__)
        with InsertionPoint(module.body):
            f()
        print(module)


# CHECK-LABEL: getTileOp
# CHECK: AIEX.getTile
@constructAndPrintInModule
def getTileOp():
    iTy = IndexType.get()
    four = arith.ConstantOp(iTy, 4)
    two = arith.ConstantOp(iTy, 2)
    GetTileOp(IndexType.get(), four, two)
