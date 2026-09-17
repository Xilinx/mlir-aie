//===- runtime_vs_static_multi_tile.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-assign-bd-ids --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// Two aie.tile ops for one (0,0): static BD on %s0, runtime task on %s1. The
// runtime allocator must seed static ids by (col,row), or it reuses bd_id 0.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(0, 0)
    %sb = aie.external_buffer {sym_name = "sb"} : memref<8xi32>

    // CHECK: aie.shim_dma
    // CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
    aie.shim_dma(%s0) {
        aie.dma_start(MM2S, 0, ^bd0, ^end)
      ^bd0:
        aie.dma_bd(%sb : memref<8xi32> offset = 0 len = 4)
        aie.next_bd ^end
      ^end:
        aie.end
    }

    // CHECK: aie.runtime_sequence
    // CHECK-NOT: aie.dma_bd({{.*}}) {bd_id = 0 : i32}
    // CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32}
    aie.runtime_sequence @seq(%arg0: memref<8xi32>) {
      %t = aiex.dma_configure_task(%s1, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi32> offset = 0 len = 4)
        aie.end
      }
      aiex.dma_start_task(%t)
    }
  }
}
