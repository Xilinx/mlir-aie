//===- dma_channel_reset_for_binding.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s

// A fifo targeted by aiex.dma_channel_reset_for gets an
// aie.objectfifo_rearm_binding recording its non-shim DMA endpoints and locks,
// and the reset_for op is retargeted from @of to @of_rearm. The binding is what
// keeps the core/mem channels + locks resolvable after the fifo op is erased
// and its @of uses are rewritten to @of_shim_alloc.

// Producer (0,2) MM2S ch0 -> dir 1; consumer (0,4) S2MM ch0 -> dir 0. Locks are
// the producer's pair then the consumer split fifo's pair, all init {2,0}.

// The reset_for op (inside the runtime sequence) prints before the binding,
// which is emitted at the end of the device body.
// CHECK-LABEL: @reset_for_binding
// CHECK-DAG: %[[T02:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[T04:.*]] = aie.tile(0, 4)
// CHECK: aiex.dma_channel_reset_for(@of_rearm)
// CHECK-NOT: aiex.dma_channel_reset_for(@of)
// CHECK: aie.objectfifo_rearm_binding @of_rearm channels(%[[T02]], %[[T04]] : index, index) locks({{.*}} : index, index, index, index) {channel_dirs = array<i32: 1, 0>, channel_indices = array<i32: 0, 0>, lock_inits = array<i32: 2, 0, 2, 0>}
module @reset_for_binding {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    aie.objectfifo @of (%t02, {%t04}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of)
    }
  }
}

// -----

// On demand: a design with no dma_channel_reset_for gets no binding, so the
// transform output is unchanged for every existing design.
// CHECK-LABEL: @no_rearm
// CHECK-NOT: aie.objectfifo_rearm_binding
module @no_rearm {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    aie.objectfifo @of (%t02, {%t04}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
  }
}

// -----

// A shim producer endpoint is skipped (the host re-pushes it): only the
// consumer core S2MM channel is recorded, so channels has one entry (dir 0).
// The shim producer's locks are also skipped (host-managed) -- only the two core
// consumer locks are re-armed, not the shim prod/cons locks.
// CHECK-LABEL: @shim_producer
// CHECK: %[[C:.*]] = aie.tile(0, 2)
// CHECK: aiex.dma_channel_reset_for(@of_in_rearm)
// CHECK: aie.objectfifo_rearm_binding @of_in_rearm channels(%[[C]] : index) locks(%[[PL:.*]], %[[CL:.*]] : index, index) {channel_dirs = array<i32: 0>, channel_indices = array<i32: 0>, lock_inits = array<i32: 2, 0>}
module @shim_producer {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @of_in (%t00, {%t02}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of_in)
    }
  }
}

// -----

// A memory-adjacent (non-split, shared-memory) fifo has no DMA channel, so the
// binding re-arms its locks only (empty channels). Without this the reset_for
// would fall through to the shim-alloc rewrite and dangle at @adj_shim_alloc.
// CHECK-LABEL: @adjacent
// CHECK: aiex.dma_channel_reset_for(@adj_rearm)
// CHECK: aie.objectfifo_rearm_binding @adj_rearm channels() locks(%{{.*}}, %{{.*}} : index, index) {channel_dirs = array<i32>, channel_indices = array<i32>, lock_inits = array<i32: 2, 0>}
module @adjacent {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @adj (%t02, {%t03}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@adj)
    }
  }
}

// -----

// The binding symbol is made unique: here @of_rearm is already taken by a lock,
// so the binding is emitted as @of_rearm_0 and the reset_for is retargeted to it
// instead of colliding.
// CHECK-LABEL: @rearm_name_collision
// CHECK: aiex.dma_channel_reset_for(@of_rearm_0)
// CHECK: aie.objectfifo_rearm_binding @of_rearm_0 channels
module @rearm_name_collision {
  aie.device(npu2) {
    %t02 = aie.tile(0, 2)
    %t04 = aie.tile(0, 4)
    %junk = aie.lock(%t02, 5) {init = 0 : i32, sym_name = "of_rearm"}
    aie.objectfifo @of (%t02, {%t04}, 2 : i32) : !aie.objectfifo<memref<64xi32>>
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@of)
    }
  }
}
