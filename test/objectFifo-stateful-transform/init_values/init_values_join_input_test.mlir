//===- init_values_join_input_test.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @init_join_input {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[SHIM_TILE:.*]] = aie.tile(1, 0)
// CHECK:     %[[MEM_TILE:.*]] = aie.tile(1, 1)
// CHECK:     %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:     %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:     %[[OF2_BUFF_0:.*]] = aie.buffer(%[[MEM_TILE]]) {sym_name = "of2_buff_0"} : memref<8xi32>
// CHECK-DAG:     %[[OF2_BUFF_1:.*]] = aie.buffer(%[[MEM_TILE]]) {sym_name = "of2_buff_1"} : memref<8xi32>
// CHECK-DAG:     %[[OF2_PROD_LOCK_0:.*]] = aie.lock(%[[MEM_TILE]], 0) {init = 2 : i32, sym_name = "of2_prod_lock_0"}
// CHECK-DAG:     %[[OF2_CONS_LOCK_0:.*]] = aie.lock(%[[MEM_TILE]], 1) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK-DAG:     %[[OF2_PROD_LOCK_1:.*]] = aie.lock(%[[MEM_TILE]], 2) {init = 2 : i32, sym_name = "of2_prod_lock_1"}
// CHECK-DAG:     %[[OF2_CONS_LOCK_1:.*]] = aie.lock(%[[MEM_TILE]], 3) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK-DAG:     %[[OF1_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of1_buff_0"} : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK-DAG:     %[[OF1_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of1_buff_1"} : memref<2x2xi32> = dense<{{\[}}[4, 5], [6, 7]]>
// CHECK-DAG:     %[[OF0_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK-DAG:     %[[OF0_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[}}[4, 5], [6, 7]]>
// Source-side locks skipped for both static-init cycling chains (of0 and
// of1 — see init_values_no_link_skips_source_locks.mlir for rationale).
// The link source qualifies because `getOptionalLinkOp` returns the
// downstream link op, not an upstream one for the source side.
// CHECK-NOT: sym_name = "of0_prod_lock_0"
// CHECK-NOT: sym_name = "of0_cons_lock_0"
// CHECK-NOT: sym_name = "of1_prod_lock_0"
// CHECK-NOT: sym_name = "of1_cons_lock_0"
// CHECK-DAG:     aie.flow(%[[TILE_1_2]], DMA : 0, %[[MEM_TILE]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[TILE_2_3]], DMA : 0, %[[MEM_TILE]], DMA : 1)
// CHECK-DAG:     aie.flow(%[[MEM_TILE]], DMA : 0, %[[SHIM_TILE]], DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%[[TILE_1_2]]) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.dma_bd(%[[OF0_BUFF_0]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.dma_bd(%[[OF0_BUFF_1]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%[[MEM_TILE]]) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<8xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<8xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<8xi32>, 4, 4)
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<8xi32>, 4, 4)
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:
// CHECK:       %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb11)
// CHECK:     ^bb7:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<8xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<8xi32>, 4, 4)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb9:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<8xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<8xi32>, 4, 4)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb11:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%[[TILE_2_3]]) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.dma_bd(%[[OF1_BUFF_0]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.dma_bd(%[[OF1_BUFF_1]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of2_shim_alloc(%[[SHIM_TILE]], S2MM, 0)
// CHECK:   }
// CHECK: }

module @init_join_input {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>, 
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
    aie.objectfifo @of1 (%tile23, {%tile11}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>, 
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
    aie.objectfifo @of2 (%tile11, {%tile10}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 4] [])
 }
}
