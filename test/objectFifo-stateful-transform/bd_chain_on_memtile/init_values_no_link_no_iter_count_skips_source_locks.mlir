//===- init_values_no_link_no_iter_count_skips_source_locks.mlir *- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Sibling of init_values_no_link_skips_source_locks.mlir, but with no
// iter_count attribute — the BD chain self-loops infinitely via
// next_bd(self) rather than restarting via the channel's task_count.
// Same correctness argument: buffers are pre-populated, nothing on the
// source side ever produces new data, and downstream back-pressure is
// handled by the DMA stream. Source-side locks would deadlock on the
// second pass and must be skipped.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module @init_values_no_link_no_iter_count_skips_source_locks {
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %[[MEM_TILE:.*]] = aie.tile(0, 1)
// CHECK:     %[[CT:.*]] = aie.tile(0, 2)
// CHECK-NOT: aie.lock(%[[MEM_TILE]]
// CHECK-NOT: sym_name = "of_prod_lock_
// CHECK-NOT: sym_name = "of_cons_lock_
// CHECK:     %{{.*}} = aie.memtile_dma(%[[MEM_TILE]]) {
// CHECK:       %{{.*}} = aie.dma_start(MM2S
// CHECK-NEXT:  ^bb1:
// CHECK-NOT:     aie.use_lock
// CHECK:         aie.dma_bd
// CHECK-NOT:     aie.use_lock
// CHECK:         aie.next_bd

module @init_values_no_link_no_iter_count_skips_source_locks {
  aie.device(npu1_1col) {
    %mem_tile = aie.tile(0, 1)
    %ct = aie.tile(0, 2)
    aie.objectfifo @of(%mem_tile, {%ct}, [2 : i32, 1 : i32])
      : !aie.objectfifo<memref<16xi32>>
      = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
  }
}
