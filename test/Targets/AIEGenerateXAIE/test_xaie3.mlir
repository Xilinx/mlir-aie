//===- test_xaie3.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test acquire with '1'.   Single BD.
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,1),XAie_LockInit(0,0)));

module @test_xaie3 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %t44 = aie.tile(4, 4)

  %buf33_0 = aie.buffer(%t33) { address = 0 : i32, sym_name = "buf33_0" }: memref<256xi32>

  %l33_0 = aie.lock(%t33, 0)

  %m33 = aie.mem(%t33) {
      %srcDma = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%l33_0, Acquire, %c1_ul1)
      aie.dma_bd(%buf33_0 : memref<256xi32> offset = 0 len = 256)
      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%l33_0, Release, %c0_ul2)
      aie.next_bd ^end
    ^end:
      aie.end
  }
 }
}
