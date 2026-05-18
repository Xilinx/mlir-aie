//===- init_values_no_link_skips_source_locks.mlir -------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// An aie.objectfifo with init_values on a MemTile producer and no upstream
// link has no producer to refill the source-side locks between BD-chain
// iterations. The lowering must omit the source-side use_lock ops so the
// chain can be restarted by the channel's task_count (= iter_count - 1)
// without deadlocking on the second pass. The downstream consumer's locks
// are unaffected and the DMA stream provides flow control.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module @init_values_no_link_skips_source_locks {
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %[[MEM_TILE:.*]] = aie.tile(0, 1)
// CHECK:     %[[CT:.*]] = aie.tile(0, 2)
// CHECK:     %{{.*}} = aie.memtile_dma(%[[MEM_TILE]]) {
// CHECK:       %{{.*}} = aie.dma_start(MM2S, 0, ^bb1, ^bb{{[0-9]+}}, repeat_count = 3)
// CHECK-NEXT:  ^bb1:
// CHECK-NOT:     aie.use_lock
// CHECK:         aie.dma_bd
// CHECK-NOT:     aie.use_lock
// CHECK:         aie.next_bd
// CHECK-NEXT:  ^bb2:
// CHECK-NOT:     aie.use_lock
// CHECK:         aie.dma_bd
// CHECK-NOT:     aie.use_lock
// CHECK:         aie.next_bd
// CHECK-NEXT:  ^bb3:
// CHECK:         aie.end

module @init_values_no_link_skips_source_locks {
  aie.device(npu1_1col) {
    %mem_tile = aie.tile(0, 1)
    %ct = aie.tile(0, 2)
    aie.objectfifo @of(%mem_tile, {%ct}, [2 : i32, 1 : i32]) {iter_count = 4 : i32}
      : !aie.objectfifo<memref<16xi32>>
      = [dense<0> : memref<16xi32>, dense<1> : memref<16xi32>]
  }
}
