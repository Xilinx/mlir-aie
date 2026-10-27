//===- bad_axcache.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file %s 2>&1 | FileCheck %s

module {
    %t21 = aie.tile(2, 2)
    aie.buffer @buf21_0(%t21) : memref<7168xi32>
    aie.lock @l21_0(%t21, 0)
    %m21 = aie.mem(%t21) {
      %srcDma = aie.dma_start(S2MM, 0, ^bd0, ^end)
        ^bd0:
        // CHECK: AxCACHE is only supported in Shim NOC tiles that are connected to the memory-mapped NOC.
        %buf21_0 = aie.get_buffer @buf21_0 : memref<7168xi32>
        aie.dma_bd(%buf21_0 : memref<7168xi32> offset = 0 len = 7168){ axcache = 2 : i32 }
        aie.next_bd ^end
      ^end:
        aie.end
    }
}

// -----

module {
 aie.device(xcvc1902) {
  aie.external_buffer @buf : memref<32x32xi32>
  %tile70 = aie.tile(7, 0)
  aie.lock @lock70(%tile70, 0)
  %shimdma70 = aie.shim_dma(%tile70)  {
    aie.dma_start(MM2S, 0, ^bb1, ^bb2)
  ^bb1:
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(@lock70, Acquire, %c1_ul1)
    // CHECK: error{{.*}}'aie.dma_bd' op attribute 'axcache' failed to satisfy constraint
    %buf = aie.get_buffer @buf : memref<32x32xi32>
    aie.dma_bd(%buf : memref<32x32xi32> offset = 0 len = 1024) { axcache = 16 : i32 }
    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(@lock70, Release, %c0_ul2)
    aie.next_bd ^bb1
  ^bb2:
    aie.end
  }
 }
}
