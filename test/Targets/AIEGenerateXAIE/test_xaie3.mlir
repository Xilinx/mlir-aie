//===- test_xaie3.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
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
      aie.use_lock(%l33_0, Acquire, 1)
      aie.dma_bd(%buf33_0 : memref<256xi32>) { len = 256 : i32 }
      aie.use_lock(%l33_0, Release, 0)
      aie.next_bd ^end
    ^end:
      aie.end
  }
 }
}
