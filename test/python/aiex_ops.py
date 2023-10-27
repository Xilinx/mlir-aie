# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %PYTHON %s | FileCheck %s

import aie
from aie.ir import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects import arith

def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        aie.dialects.aiex.register_dialect(ctx)
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
