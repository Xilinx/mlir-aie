//===- test_xaie3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-translate --aie-generate-xaie %s | FileCheck %s

// Test acquire with '1'.   Single BD.
// CHECK: XAie_DmaDesc [[bd0:.*]];
// CHECK: __mlir_aie_try(XAie_DmaSetLock(&([[bd0]]), XAie_LockInit(0,1),XAie_LockInit(0,0)));

module @test_xaie3 {
 AIE.device(xcvc1902) {
  %t33 = AIE.tile(3, 3)
  %t44 = AIE.tile(4, 4)

  %buf33_0 = AIE.buffer(%t33) { address = 0, sym_name = "buf33_0" }: memref<256xi32>

  %l33_0 = AIE.lock(%t33, 0)

  %m33 = AIE.mem(%t33) {
      %srcDma = AIE.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      AIE.use_lock(%l33_0, Acquire, 1)
      AIE.dma_bd(<%buf33_0 : memref<256xi32>, 0, 256>, A)
      AIE.use_lock(%l33_0, Release, 0)
      AIE.next_bd ^end
    ^end:
      AIE.end
  }
 }
}
