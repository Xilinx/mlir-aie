//===- bad-chain-exceeds-partition.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-lower-dynamic-bd-pool --verify-diagnostics %s

// Every BD of a chain holds a pool id at once, so a chain longer than its
// channel's partition could never be allocated. A shim channel draws from 16
// ids; this chain has 17 BDs.

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @too_long(%arg0: memref<1088xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      // expected-error@+1 {{chains 17 buffer descriptors, but channel 0 of tile (0,0) draws from a pool of only 16 ids}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 0 len = 64)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 64 len = 64)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 128 len = 64)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 192 len = 64)
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 256 len = 64)
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 320 len = 64)
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 384 len = 64)
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 448 len = 64)
        aie.next_bd ^bd8
      ^bd8:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 512 len = 64)
        aie.next_bd ^bd9
      ^bd9:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 576 len = 64)
        aie.next_bd ^bd10
      ^bd10:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 640 len = 64)
        aie.next_bd ^bd11
      ^bd11:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 704 len = 64)
        aie.next_bd ^bd12
      ^bd12:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 768 len = 64)
        aie.next_bd ^bd13
      ^bd13:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 832 len = 64)
        aie.next_bd ^bd14
      ^bd14:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 896 len = 64)
        aie.next_bd ^bd15
      ^bd15:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 960 len = 64)
        aie.next_bd ^bd16
      ^bd16:
        aie.dma_bd(%arg0 : memref<1088xi32> offset = 1024 len = 64)
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
