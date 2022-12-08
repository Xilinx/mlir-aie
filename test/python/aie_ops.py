# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: python %s | FileCheck %s

import aie
from aie.mlir.ir import *
from aie.dialects.aie import *

def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        aie.dialects.aie.register_dialect(ctx)
        module = Module.create()
        print("\nTEST:", f.__name__)
        with InsertionPoint(module.body):
            f()
        print(module)

# CHECK-LABEL: tileOp
# CHECK: AIE.tile(0, 0)
@constructAndPrintInModule
def tileOp():
    iTy = IntegerType.get_signless(32)
    row = IntegerAttr.get(iTy, 0)
    col = IntegerAttr.get(iTy, 0)
    t = TileOp(IndexType.get(), col, row)

# CHECK-LABEL: coreOp
# CHECK: %[[VAL_1:.*]] = AIE.tile(1, 1)
# CHECK: %[[VAL_2:.*]] = AIE.core(%[[VAL_1]]) {
# CHECK:   AIE.end
# CHECK: }
@constructAndPrintInModule
def coreOp():
    iTy = IntegerType.get_signless(32)
    row = IntegerAttr.get(iTy, 1)
    col = IntegerAttr.get(iTy, 1)
    t = TileOp(IndexType.get(), col, row)
    c = CoreOp(IndexType.get(), t)
    bb = Block.create_at_start(c.body)
    with InsertionPoint(bb):
        EndOp()

# CHECK-LABEL: memOp
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: %[[VAL_2:.*]] = AIE.mem(%[[VAL_1]]) {
# CHECK:   AIE.end
# CHECK: }
@constructAndPrintInModule
def memOp():
    iTy = IntegerType.get_signless(32)
    row = IntegerAttr.get(iTy, 2)
    col = IntegerAttr.get(iTy, 2)
    t = TileOp(IndexType.get(), col, row)
    m = MemOp(IndexType.get(), t)
    bb = Block.create_at_start(m.body)
    with InsertionPoint(bb):
        EndOp()
