//===- lock_packing_aie1_parity.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: not aie-opt --aie-objectFifo-stateful-transform %t.mlir 2>&1 | FileCheck %s

// Each tile has 15 locks free after its user lock and the pools take 6 less
// than the 135. Every pool takes an even count but @f7's 3, so a tile without
// @f7 wastes at least 1 lock; @f7 goes on one tile, so 8 tiles waste 8. The
// packing search shows this rather than running out of steps.
// CHECK: could not place locks within tile lock capacity
// CHECK-NOT: could not decide

module @aie1_pack_parity {
  aie.device(xcvc1902) {
    %t0 = aie.tile(5, 1)
    %t1 = aie.tile(5, 2)
    %t2 = aie.tile(5, 3)
    %t3 = aie.tile(6, 1)
    %t4 = aie.tile(6, 2)
    %t5 = aie.tile(6, 3)
    %t6 = aie.tile(7, 1)
    %t7 = aie.tile(7, 2)
    %t8 = aie.tile(7, 3)
    %l0 = aie.lock(%t0, 15)
    %l1 = aie.lock(%t1, 15)
    %l2 = aie.lock(%t2, 15)
    %l3 = aie.lock(%t3, 15)
    %l4 = aie.lock(%t4, 15)
    %l5 = aie.lock(%t5, 15)
    %l6 = aie.lock(%t6, 15)
    %l7 = aie.lock(%t7, 15)
    %l8 = aie.lock(%t8, 15)
    aie.objectfifo @f0(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t3, {%t3}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t8, {%t8}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t4, {%t4}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t3, {%t3}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t7, {%t7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t5, {%t5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t8, {%t8}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t0, {%t0}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t8, {%t8}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t0, {%t0}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t8, {%t8}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f34(%t2, {%t2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f35(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f36(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f37(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f38(%t4, {%t4}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f39(%t2, {%t2}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f40(%t7, {%t7}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f41(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f42(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f43(%t4, {%t4}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
