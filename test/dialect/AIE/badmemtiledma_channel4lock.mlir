//===- memtiledma.mlir -----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --canonicalize %s 2>&1 | FileCheck %s
// CHECK: error{{.*}}'AIE.use_lock' op is reachable from DMA channel 4 and attempts to access a non-local lock
// CHECK: note: channel
// CHECK:     AIE.dma_start("MM2S", 4, ^bd0, ^dma1)
// CHECK: note: lock
// CHECK:   %lock2 = AIE.lock(%t2, 3) { sym_name = "lock2" }

AIE.device(xcve2802) {
  %t1 = AIE.tile(1, 1)
  %t2 = AIE.tile(2, 1)
  %buf1 = AIE.buffer(%t1) : memref<256xi32>
  %lock1 = AIE.lock(%t1, 3) { sym_name = "lock1" }
  %lock2 = AIE.lock(%t2, 3) { sym_name = "lock2" }
  %mem = AIE.memtile_dma(%t1) {
    AIE.dma_start("MM2S", 4, ^bd0, ^dma1)
    ^dma1:
    AIE.dma_start("MM2S", 1, ^bd1, ^dma1)
    ^bd0:
      AIE.use_lock(%lock2, "Acquire", 1)
      AIE.dma_bd(%buf1 : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd2
    ^bd1:
      AIE.use_lock(%lock1, "Acquire", 1)
      AIE.dma_bd(%buf1 : memref<256xi32>, 0, 256)
      AIE.next_bd ^bd2
    ^bd2:
      AIE.end
  }
}
