//===- dma_bd_iteration.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// iteration attrs parse and print on a memtile dma_bd + n-dim access.
module {
  aie.device(npu2) {
    %t = aie.tile(0, 1)
    aie.buffer @b(%t) : memref<256xi32>
    %m = aie.memtile_dma(%t) {
      %s = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      // CHECK: aie.dma_bd
      // CHECK-SAME: iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2>
      %b = aie.get_buffer @b : memref<256xi32>
      aie.dma_bd(%b : memref<256xi32> offset = 0 len = 64 sizes = [4, 4] strides = [4, 1]) { iteration = #aie.bd_iteration<size = 4, stride = 16, current = 2> }
      aie.next_bd ^end
    ^end:
      aie.end
    }
  }
}
