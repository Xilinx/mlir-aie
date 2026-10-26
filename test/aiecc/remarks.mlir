//===- remarks.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Remarks say what the compiler chose. aiecc prints them with --verbose, or
// once a compile fails; a compile that succeeds prints none.

// reclaim.mlir's @j1: with --reclaim-runtime-bds, a task takes back an earlier
// one's BD id behind a poll aie-assign-runtime-sequence-bd-ids inserts and
// reports as a remark.
// RUN: %aiecc -n --tmpdir %t.quiet --get-npu-insts --npu-insts-name=%t.quiet.bin --reclaim-runtime-bds %s 2>&1 | FileCheck %s --allow-empty --check-prefix=QUIET
// RUN: %aiecc -n -v --tmpdir %t.verbose --get-npu-insts --npu-insts-name=%t.verbose.bin --reclaim-runtime-bds %s 2>&1 | FileCheck %s --check-prefix=VERBOSE

// QUIET-NOT:   remark:
// QUIET-NOT:   the task it waits for
// VERBOSE:     remark: tile (0,0) has no free buffer descriptor id
// VERBOSE:     note: the task it waits for

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    aie.runtime_sequence @j1(%arg0: memref<64xi32>) {
      %t0 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.end
      }
      aiex.dma_start_task(%t1)
      %big = aiex.dma_configure_task(%shim, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd1
      ^bd1:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd2
      ^bd2:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd3
      ^bd3:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd4
      ^bd4:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd5
      ^bd5:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd6
      ^bd6:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd7
      ^bd7:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd8
      ^bd8:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd9
      ^bd9:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd10
      ^bd10:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd11
      ^bd11:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd12
      ^bd12:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.next_bd ^bd13
      ^bd13:
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.end
      }
      aiex.dma_start_task(%big)
      // expected-remark@+1 {{queue behind an earlier task is short enough to prove it finished and takes back its ids 0}}
      %new = aiex.dma_configure_task(%shim, S2MM, 0) {
        aie.dma_bd(%arg0 : memref<64xi32> offset = 0 len = 8)
        aie.end
      }
      aiex.dma_start_task(%new)
    }
  }
}
