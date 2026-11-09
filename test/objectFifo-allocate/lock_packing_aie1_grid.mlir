//===- lock_packing_aie1_grid.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Each tile's 16 locks are split into pools of depths 2 to 7, each owned by
// that tile or a vertical neighbour, so a packing that fills all 128 locks
// exists. Placing the 34 pools one at a time, largest first, takes some 271k
// steps to find it, past the search budget; filling a tile at a time does not.
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
    aie.core(%t0) {
      %p2 = aie.objectfifo.acquire @f2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Produce, 1)
      %c2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %p3 = aie.objectfifo.acquire @f3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Produce, 1)
      %c3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %p6 = aie.objectfifo.acquire @f6(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Produce, 1)
      %c6 = aie.objectfifo.acquire @f6(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f6(Consume, 1)
      %p10 = aie.objectfifo.acquire @f10(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Produce, 1)
      %c10 = aie.objectfifo.acquire @f10(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f10(Consume, 1)
      aie.end
    }
    aie.core(%t1) {
      %p0 = aie.objectfifo.acquire @f0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Produce, 1)
      %c0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %p1 = aie.objectfifo.acquire @f1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Produce, 1)
      %c1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %p4 = aie.objectfifo.acquire @f4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Produce, 1)
      %c4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %p5 = aie.objectfifo.acquire @f5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Produce, 1)
      %c5 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      %p8 = aie.objectfifo.acquire @f8(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Produce, 1)
      %c8 = aie.objectfifo.acquire @f8(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f8(Consume, 1)
      %p9 = aie.objectfifo.acquire @f9(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Produce, 1)
      %c9 = aie.objectfifo.acquire @f9(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f9(Consume, 1)
      %p12 = aie.objectfifo.acquire @f12(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Produce, 1)
      %c12 = aie.objectfifo.acquire @f12(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f12(Consume, 1)
      aie.end
    }
    aie.core(%t2) {
      %p7 = aie.objectfifo.acquire @f7(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Produce, 1)
      %c7 = aie.objectfifo.acquire @f7(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f7(Consume, 1)
      %p11 = aie.objectfifo.acquire @f11(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Produce, 1)
      %c11 = aie.objectfifo.acquire @f11(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f11(Consume, 1)
      %p13 = aie.objectfifo.acquire @f13(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Produce, 1)
      %c13 = aie.objectfifo.acquire @f13(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f13(Consume, 1)
      aie.end
    }
    aie.core(%t3) {
      %p14 = aie.objectfifo.acquire @f14(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Produce, 1)
      %c14 = aie.objectfifo.acquire @f14(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f14(Consume, 1)
      %p15 = aie.objectfifo.acquire @f15(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Produce, 1)
      %c15 = aie.objectfifo.acquire @f15(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f15(Consume, 1)
      %p16 = aie.objectfifo.acquire @f16(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Produce, 1)
      %c16 = aie.objectfifo.acquire @f16(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f16(Consume, 1)
      %p17 = aie.objectfifo.acquire @f17(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Produce, 1)
      %c17 = aie.objectfifo.acquire @f17(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f17(Consume, 1)
      aie.end
    }
    aie.core(%t4) {
      %p18 = aie.objectfifo.acquire @f18(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Produce, 1)
      %c18 = aie.objectfifo.acquire @f18(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f18(Consume, 1)
      %p19 = aie.objectfifo.acquire @f19(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Produce, 1)
      %c19 = aie.objectfifo.acquire @f19(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f19(Consume, 1)
      aie.end
    }
    aie.core(%t5) {
      %p20 = aie.objectfifo.acquire @f20(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Produce, 1)
      %c20 = aie.objectfifo.acquire @f20(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f20(Consume, 1)
      %p21 = aie.objectfifo.acquire @f21(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Produce, 1)
      %c21 = aie.objectfifo.acquire @f21(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f21(Consume, 1)
      %p22 = aie.objectfifo.acquire @f22(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Produce, 1)
      %c22 = aie.objectfifo.acquire @f22(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f22(Consume, 1)
      %p25 = aie.objectfifo.acquire @f25(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Produce, 1)
      %c25 = aie.objectfifo.acquire @f25(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f25(Consume, 1)
      %p28 = aie.objectfifo.acquire @f28(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Produce, 1)
      %c28 = aie.objectfifo.acquire @f28(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f28(Consume, 1)
      aie.end
    }
    aie.core(%t6) {
      %p23 = aie.objectfifo.acquire @f23(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Produce, 1)
      %c23 = aie.objectfifo.acquire @f23(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f23(Consume, 1)
      %p24 = aie.objectfifo.acquire @f24(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Produce, 1)
      %c24 = aie.objectfifo.acquire @f24(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f24(Consume, 1)
      %p26 = aie.objectfifo.acquire @f26(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Produce, 1)
      %c26 = aie.objectfifo.acquire @f26(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f26(Consume, 1)
      %p27 = aie.objectfifo.acquire @f27(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Produce, 1)
      %c27 = aie.objectfifo.acquire @f27(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f27(Consume, 1)
      %p29 = aie.objectfifo.acquire @f29(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Produce, 1)
      %c29 = aie.objectfifo.acquire @f29(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f29(Consume, 1)
      %p30 = aie.objectfifo.acquire @f30(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Produce, 1)
      %c30 = aie.objectfifo.acquire @f30(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f30(Consume, 1)
      aie.end
    }
    aie.core(%t7) {
      %p31 = aie.objectfifo.acquire @f31(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Produce, 1)
      %c31 = aie.objectfifo.acquire @f31(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f31(Consume, 1)
      %p32 = aie.objectfifo.acquire @f32(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Produce, 1)
      %c32 = aie.objectfifo.acquire @f32(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f32(Consume, 1)
      %p33 = aie.objectfifo.acquire @f33(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Produce, 1)
      %c33 = aie.objectfifo.acquire @f33(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f33(Consume, 1)
      aie.end
    }
  }
}
