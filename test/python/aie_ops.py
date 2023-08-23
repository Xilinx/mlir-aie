# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: python3 %s | FileCheck %s

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

# CHECK-LABEL: deviceOp
# CHECK: AIE.device
@constructAndPrintInModule
def deviceOp():
    iTy = IntegerType.get_signless(32)
    i = IntegerAttr.get(iTy, 1)
    dev = DeviceOp(IntegerAttr.get(iTy, 1))
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        EndOp()

# CHECK-LABEL: objFifo
# CHECK: %[[VAL_0:.*]] = AIE.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = AIE.tile(2, 2)
# CHECK: AIE.objectFifo @of0(%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<12xf16>>
@constructAndPrintInModule
def objFifo():
    iTy = IntegerType.get_signless(32)
    one = IntegerAttr.get(iTy, 1)
    dev = DeviceOp(one)
    bb = Block.create_at_start(dev.bodyRegion)
    with InsertionPoint(bb):
        two = IntegerAttr.get(iTy, 2)
        six = IntegerAttr.get(iTy, 6)
        tile0 = TileOp(IndexType.get(), six, six)
        tile1 = TileOp(IndexType.get(), two, two)
        dtype = F16Type.get()
        memTy = MemRefType.get((12,), dtype)
        ofTy = ObjectFifoType.get(memTy)
        ObjectFifoCreateOp("of0", tile0, tile1, two, TypeAttr.get(ofTy))
        EndOp()
