//===- reserve_static_bd_ids.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The runtime allocator must not reuse a bd_id the static tile program already
// assigned on the same tile.

// RUN: aie-opt --aie-assign-bd-ids --aie-materialize-bd-chains --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// CHECK: aiex.dma_configure_task
// CHECK: aie.dma_bd(%{{.*}} : memref<8xi32>) {bd_id = 1 : i32}
aie.device(npu2) {
  %t = aie.tile(0, 0)
  %buf = aie.external_buffer {sym_name = "b"} : memref<8xi32>
  aie.shim_dma(%t) {
    %0 = aie.dma_start(MM2S, 0, ^bd, ^end)
  ^bd:
    aie.dma_bd(%buf : memref<8xi32>) {bd_id = 0 : i32}
    aie.next_bd ^end
  ^end:
    aie.end
  }
  aie.runtime_sequence(%a: memref<8xi32>) {
    %task = aiex.dma_configure_task(%t, S2MM, 1) {
      aie.dma_bd(%a : memref<8xi32>)
      aie.end
    }
  }
}
