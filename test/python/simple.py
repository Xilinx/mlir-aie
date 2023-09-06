# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# module {
#   %0 = AIE.tile(1, 4)
#   %1 = AIE.buffer(%0) : memref<256xi32>
#   %2 = AIE.core(%0) {
#   ^bb0(%arg0: index):
#     %c14_i32 = arith.constant 14 : i32
#     %c3 = arith.constant 3 : index
#     memref.store %c14_i32, %1[%c3] : memref<256xi32>
#     AIE.end
#   }
#}

import aie
from aie.mlir.ir import *
from aie.dialects import aie as aiedialect
from aie.mlir.dialects import arith
from aie.mlir.dialects import memref

with Context() as ctx, Location.unknown():
  aiedialect.register_dialect(ctx)
  module = Module.create()
  with InsertionPoint(module.body):
    int_ty = IntegerType.get_signless(32)
    idx_ty = IndexType.get()
    memRef_ty = MemRefType.get((256,), int_ty)
    
    T = aiedialect.TileOp(idx_ty, IntegerAttr.get(int_ty, 1), IntegerAttr.get(int_ty, 4))
    buff = aiedialect.BufferOp(memRef_ty, T)

    C = aiedialect.CoreOp(idx_ty, T)
    C.body.blocks.append(idx_ty)
    with InsertionPoint(C.body.blocks[0]):
        val = arith.ConstantOp(int_ty, IntegerAttr.get(int_ty, 14))
        idx = arith.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, 3))
        memref.StoreOp(val, buff, idx)
        aiedialect.EndOp()
      
print(module)
