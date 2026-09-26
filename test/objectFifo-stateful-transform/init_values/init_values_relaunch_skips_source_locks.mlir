//===- init_values_relaunch_skips_source_locks.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform="skip-verify=true" --aie-objectFifo-unroll %s | FileCheck %s

// A constant MemTile fifo read once per launch (no iter_count) must not get
// source locks. Counting locks armed with depth * repeat tokens run dry after
// the first launch, and the consumer of the next launch hangs (mobilenet's
// post_L1/post_L2 weights did). The consumer's own locks are kept.

// CHECK-LABEL: module @relaunch
// CHECK-DAG:     %[[MEM_TILE:.*]] = aie.tile(1, 1)
// CHECK-NOT:     sym_name = "wts_prod_lock_0"
// CHECK-NOT:     sym_name = "wts_cons_lock_0"
// CHECK:         aie.lock({{.*}}) {init = 1 : i32, sym_name = "wts_cons_prod_lock_0"}
// CHECK:         aie.memtile_dma(%[[MEM_TILE]]) {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 6)
// CHECK-NEXT:    ^bb1:
// CHECK-NOT:       aie.use_lock
// CHECK:           aie.dma_bd
// CHECK-NOT:       aie.use_lock
// CHECK:           aie.next_bd ^bb1
module @relaunch {
  aie.device(npu2) {
    %mem_tile = aie.tile(1, 1)
    %ct = aie.tile(1, 2)
    aie.objectfifo @wts(%mem_tile, {%ct}, 1 : i32) {repeat_count = 7 : i32}
      : !aie.objectfifo<memref<16xi32>> = [dense<3> : memref<16xi32>]
  }
}

// -----

// Named by aiex.dma_channel_reset_for, the same fifo keeps its locks: the
// reset re-arms them to their initial counts at the start of every launch.

// CHECK-LABEL: module @rearmed
// CHECK:         aie.lock({{.*}}) {init = 0 : i32, sym_name = "wts_prod_lock_0"}
// CHECK:         aie.lock({{.*}}) {init = 7 : i32, sym_name = "wts_cons_lock_0"}
// CHECK:         aie.memtile_dma
// CHECK:           aie.use_lock({{.*}}, AcquireGreaterEqual
module @rearmed {
  aie.device(npu2) {
    %mem_tile = aie.tile(1, 1)
    %ct = aie.tile(1, 2)
    aie.objectfifo @wts(%mem_tile, {%ct}, 1 : i32) {repeat_count = 7 : i32}
      : !aie.objectfifo<memref<16xi32>> = [dense<3> : memref<16xi32>]
    aie.runtime_sequence() {
      aiex.dma_channel_reset_for(@wts)
    }
  }
}
