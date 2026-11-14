//===- bad_dma_configure_task_alloc.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// A task's alloc says which shim allocation's transfers it carries, which BD
// id assignment credits by; it must name the channel the task runs on.
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @out(%t, S2MM, 0)
  aie.runtime_sequence(%c: memref<64xi32>) {
    // expected-error@+1 {{runs on another channel than its alloc @out}}
    %task = aiex.dma_configure_task(%t, S2MM, 1) {
      aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {alloc = @out}
  }
}

// -----

aie.device(npu2) {
  %t = aie.tile(0, 0)
  %c2 = aie.tile(0, 2)
  %buf = aie.buffer(%c2) {sym_name = "notalloc"} : memref<16xi32>
  aie.runtime_sequence(%c: memref<64xi32>) {
    // expected-error@+1 {{alloc @notalloc names no aie.shim_dma_allocation}}
    %task = aiex.dma_configure_task(%t, S2MM, 0) {
      aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {alloc = @notalloc}
  }
}

// -----

// The same channel index on another tile.
aie.device(npu2) {
  %t = aie.tile(0, 0)
  %t1 = aie.tile(1, 0)
  aie.shim_dma_allocation @out(%t, S2MM, 0)
  aie.runtime_sequence(%c: memref<64xi32>) {
    // expected-error@+1 {{runs on another channel than its alloc @out}}
    %task = aiex.dma_configure_task(%t1, S2MM, 0) {
      aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {alloc = @out}
  }
}

// -----

// The same tile and channel index in the other direction.
aie.device(npu2) {
  %t = aie.tile(0, 0)
  aie.shim_dma_allocation @out(%t, S2MM, 0)
  aie.runtime_sequence(%c: memref<64xi32>) {
    // expected-error@+1 {{runs on another channel than its alloc @out}}
    %task = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%c : memref<64xi32> offset = 0 len = 64)
      aie.end
    } {alloc = @out}
  }
}
