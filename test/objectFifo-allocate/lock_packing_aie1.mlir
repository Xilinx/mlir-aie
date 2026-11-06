//===- lock_packing_aie1.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// AIE1 pools take one lock an object, so pools of different depths take
// different counts. Six pools of 6, 6, 5, 5, 5 and 5 need all 32 locks of
// (5, 2) and (5, 3); placed largest first, the last 5 finds no room, but
// 6 + 5 + 5 on each tile holds them all.
// CHECK-COUNT-16: aie.lock(%tile_5_2)
// CHECK-COUNT-16: aie.lock(%tile_5_3)

module @aie1_pack {
  aie.device(xcvc1902) {
    %x = aie.tile(5, 2)
    %y = aie.tile(5, 3)
    aie.objectfifo @f0(%x, {%x}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f1(%x, {%x}, 6 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f2(%x, {%x}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f3(%x, {%x}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f4(%x, {%x}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @f5(%x, {%x}, 5 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%x) {
      %p0 = aie.objectfifo.acquire @f0(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Produce, 1)
      %c0 = aie.objectfifo.acquire @f0(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f0(Consume, 1)
      %p1 = aie.objectfifo.acquire @f1(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Produce, 1)
      %c1 = aie.objectfifo.acquire @f1(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f1(Consume, 1)
      %p2 = aie.objectfifo.acquire @f2(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Produce, 1)
      %c2 = aie.objectfifo.acquire @f2(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f2(Consume, 1)
      %p3 = aie.objectfifo.acquire @f3(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Produce, 1)
      %c3 = aie.objectfifo.acquire @f3(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f3(Consume, 1)
      %p4 = aie.objectfifo.acquire @f4(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Produce, 1)
      %c4 = aie.objectfifo.acquire @f4(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f4(Consume, 1)
      %p5 = aie.objectfifo.acquire @f5(Produce, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Produce, 1)
      %c5 = aie.objectfifo.acquire @f5(Consume, 1) : memref<16xi32>
      aie.objectfifo.release @f5(Consume, 1)
      aie.end
    }
  }
}
