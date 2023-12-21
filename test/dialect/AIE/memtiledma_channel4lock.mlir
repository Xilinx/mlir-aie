//===- memtiledma.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// memtiles can access neighboring buffers

// CHECK-LABEL: module {
// CHECK:       }

aie.device(xcve2802) {
  %t1 = aie.tile(1, 1)
  %t2 = aie.tile(2, 1)
  %buf1 = aie.buffer(%t1) : memref<256xi32>
  %lock1 = aie.lock(%t1, 3) { sym_name = "lock1" }
  %lock2 = aie.lock(%t2, 3) { sym_name = "lock2" }
  %mem = aie.memtile_dma(%t1) {
    aie.dma_start("MM2S", 4, ^bd0, ^dma1)
    ^dma1:
    aie.dma_start("MM2S", 1, ^bd1, ^dma1)
    ^bd0:
      aie.use_lock(%lock1, "Acquire", 1)
      aie.dma_bd(%buf1 : memref<256xi32>, 0, 256)
      aie.next_bd ^bd2
    ^bd1:
      aie.use_lock(%lock2, "Acquire", 1)
      aie.dma_bd(%buf1 : memref<256xi32>, 0, 256)
      aie.next_bd ^bd2
    ^bd2:
      aie.end
  }
}
