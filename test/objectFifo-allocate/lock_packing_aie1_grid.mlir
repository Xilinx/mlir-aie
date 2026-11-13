//===- lock_packing_aie1_grid.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python %S/../../utils/generate_self_loop_cores.py %s > %t.mlir
// RUN: aie-opt --aie-objectFifo-stateful-transform %t.mlir | FileCheck %s

// Each tile's 16 locks are split into pools of depths 2 to 7, each owned by
// that tile or a vertical neighbour, so a packing that fills all 128 locks
// exists. Placing the 34 pools one at a time, largest first, takes some 271k
// steps to find it; filling a tile at a time takes under 100k.
// CHECK-COUNT-16: aie.lock(%tile_5_1)
// CHECK-COUNT-16: aie.lock(%tile_5_2)
// CHECK-COUNT-16: aie.lock(%tile_5_3)
// CHECK-COUNT-16: aie.lock(%tile_5_4)
// CHECK-COUNT-16: aie.lock(%tile_6_1)
// CHECK-COUNT-16: aie.lock(%tile_6_2)
// CHECK-COUNT-16: aie.lock(%tile_6_3)
// CHECK-COUNT-16: aie.lock(%tile_6_4)

module @aie1_pack_grid {
  aie.device(xcvc1902) {
    %t0 = aie.tile(5, 1)
    %t1 = aie.tile(5, 2)
    %t2 = aie.tile(5, 3)
    %t3 = aie.tile(5, 4)
    %t4 = aie.tile(6, 1)
    %t5 = aie.tile(6, 2)
    %t6 = aie.tile(6, 3)
    %t7 = aie.tile(6, 4)
    aie.objectfifo @f0(%t1, {%t1}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%t0, {%t0}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f6(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f7(%t2, {%t2}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f8(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f9(%t1, {%t1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f10(%t0, {%t0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f11(%t2, {%t2}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f12(%t1, {%t1}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f13(%t2, {%t2}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f14(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f15(%t3, {%t3}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f16(%t3, {%t3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f17(%t3, {%t3}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f18(%t4, {%t4}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f19(%t4, {%t4}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f20(%t5, {%t5}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f21(%t5, {%t5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f22(%t5, {%t5}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f23(%t6, {%t6}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f24(%t6, {%t6}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f25(%t5, {%t5}, 4 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f26(%t6, {%t6}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f27(%t6, {%t6}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f28(%t5, {%t5}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f29(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f30(%t6, {%t6}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f31(%t7, {%t7}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f32(%t7, {%t7}, 7 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f33(%t7, {%t7}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
