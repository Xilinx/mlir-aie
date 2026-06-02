//===- link_test_join_offsets.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 28th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @link_distribute_offsets {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %{{.*}}tile_2_0 = aie.tile(2, 0)
// CHECK:     %{{.*}}tile_2_1 = aie.tile(2, 1)
// CHECK:     %{{.*}}tile_2_2 = aie.tile(2, 2)
// CHECK:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK-DAG: %[[LINK4_BUFF_0:.*]] = aie.buffer(%{{.*}}tile_2_1) {sym_name = "link4_buff_0"} : memref<48xi32>
// CHECK-DAG: %[[LINK4_BUFF_1:.*]] = aie.buffer(%{{.*}}tile_2_1) {sym_name = "link4_buff_1"} : memref<48xi32>
// CHECK-DAG: %[[LINK4_PROD_LOCK_0:.*]] = aie.lock(%{{.*}}tile_2_1, 0) {init = 2 : i32, sym_name = "link4_prod_lock_0"}
// CHECK-DAG: %[[LINK4_CONS_LOCK_0:.*]] = aie.lock(%{{.*}}tile_2_1, 1) {init = 0 : i32, sym_name = "link4_cons_lock_0"}
// CHECK-DAG: %[[LINK4_PROD_LOCK_1:.*]] = aie.lock(%{{.*}}tile_2_1, 2) {init = 2 : i32, sym_name = "link4_prod_lock_1"}
// CHECK-DAG: %[[LINK4_CONS_LOCK_1:.*]] = aie.lock(%{{.*}}tile_2_1, 3) {init = 0 : i32, sym_name = "link4_cons_lock_1"}
// CHECK-DAG: %[[LINK4_PROD_LOCK_2:.*]] = aie.lock(%{{.*}}tile_2_1, 4) {init = 2 : i32, sym_name = "link4_prod_lock_2"}
// CHECK-DAG: %[[LINK4_CONS_LOCK_2:.*]] = aie.lock(%{{.*}}tile_2_1, 5) {init = 0 : i32, sym_name = "link4_cons_lock_2"}
// CHECK-DAG: %[[LINK3_BUFF_0:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "link3_buff_0"} : memref<12xi32>
// CHECK-DAG: %[[LINK3_BUFF_1:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "link3_buff_1"} : memref<12xi32>
// CHECK-DAG: %[[LINK3_PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_3_3, 0) {init = 2 : i32, sym_name = "link3_prod_lock_0"}
// CHECK-DAG: %[[LINK3_CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_3_3, 1) {init = 0 : i32, sym_name = "link3_cons_lock_0"}
// CHECK-DAG: %[[LINK2_BUFF_0:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "link2_buff_0"} : memref<20xi32>
// CHECK-DAG: %[[LINK2_BUFF_1:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "link2_buff_1"} : memref<20xi32>
// CHECK-DAG: %[[LINK2_PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_2_3, 0) {init = 2 : i32, sym_name = "link2_prod_lock_0"}
// CHECK-DAG: %[[LINK2_CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_2_3, 1) {init = 0 : i32, sym_name = "link2_cons_lock_0"}
// CHECK-DAG: %[[LINK1_BUFF_0:.*]] = aie.buffer(%{{.*}}tile_2_2) {sym_name = "link1_buff_0"} : memref<4x4xi32>
// CHECK-DAG: %[[LINK1_BUFF_1:.*]] = aie.buffer(%{{.*}}tile_2_2) {sym_name = "link1_buff_1"} : memref<4x4xi32>
// CHECK-DAG: %[[LINK1_PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_2_2, 0) {init = 2 : i32, sym_name = "link1_prod_lock_0"}
// CHECK-DAG: %[[LINK1_CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_2_2, 1) {init = 0 : i32, sym_name = "link1_cons_lock_0"}
// CHECK-DAG: aie.flow(%{{.*}}tile_2_2, DMA : 0, %{{.*}}tile_2_1, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_2_3, DMA : 0, %{{.*}}tile_2_1, DMA : 1)
// CHECK-DAG: aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}tile_2_1, DMA : 2)
// CHECK-DAG: aie.flow(%{{.*}}tile_2_1, DMA : 0, %{{.*}}tile_2_0, DMA : 0)
// CHECK:     %mem_2_2 = aie.mem(%{{.*}}tile_2_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK1_BUFF_0]] : memref<4x4xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LINK1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK1_BUFF_1]] : memref<4x4xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK1_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_2_1 = aie.memtile_dma(%{{.*}}tile_2_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0]] : memref<48xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1]] : memref<48xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0]] : memref<48xi32>, 16, 20)
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1]] : memref<48xi32>, 16, 20)
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:
// CHECK:       %2 = aie.dma_start(S2MM, 2, ^bb7, ^bb9)
// CHECK:     ^bb7:
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0]] : memref<48xi32>, 36, 12)
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_2]], Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1]] : memref<48xi32>, 36, 12)
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_2]], Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb9:
// CHECK:       %3 = aie.dma_start(MM2S, 0, ^bb10, ^bb16)
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0]] : memref<48xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb11
// CHECK:     ^bb11:
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0]] : memref<48xi32>, 16, 20)
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb12
// CHECK:     ^bb12:
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_0]] : memref<48xi32>, 36, 12)
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_2]], Release, 1)
// CHECK:       aie.next_bd ^bb13
// CHECK:     ^bb13:
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1]] : memref<48xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb14
// CHECK:     ^bb14:
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1]] : memref<48xi32>, 16, 20)
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb15
// CHECK:     ^bb15:
// CHECK:       aie.use_lock(%[[LINK4_CONS_LOCK_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK4_BUFF_1]] : memref<48xi32>, 36, 12)
// CHECK:       aie.use_lock(%[[LINK4_PROD_LOCK_2]], Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb16:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK2_BUFF_0]] : memref<20xi32>, 0, 20)
// CHECK:       aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LINK2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK2_BUFF_1]] : memref<20xi32>, 0, 20)
// CHECK:       aie.use_lock(%[[LINK2_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LINK3_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK3_BUFF_0]] : memref<12xi32>, 0, 12)
// CHECK:       aie.use_lock(%[[LINK3_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LINK3_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[LINK3_BUFF_1]] : memref<12xi32>, 0, 12)
// CHECK:       aie.use_lock(%[[LINK3_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @link4_shim_alloc(%shim_noc_tile_2_0, S2MM, 0)
// CHECK:   }
// CHECK: }

module @link_distribute_offsets {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
        aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

        aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
    }
}
