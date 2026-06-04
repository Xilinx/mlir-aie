//===- nd_dma_fromStream_join.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @ndDMAObjFifoAIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK-DAG:     %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK-DAG:     %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:     %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK-DAG:     %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK-DAG:     %[[OF2_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of2_cons_buff_0"} : memref<256xi32>
// CHECK-DAG:     %[[OF2_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_2_3]]) {sym_name = "of2_cons_buff_1"} : memref<256xi32>
// CHECK-DAG:     %[[OF2_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK-DAG:     %[[OF2_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_3]], 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK-DAG:     %[[OF2_BUFF_0:.*]] = aie.buffer(%[[TILE_1_1]]) {sym_name = "of2_buff_0"} : memref<256xi32>
// CHECK-DAG:     %[[OF2_BUFF_1:.*]] = aie.buffer(%[[TILE_1_1]]) {sym_name = "of2_buff_1"} : memref<256xi32>
// CHECK-DAG:     %[[OF2_PROD_LOCK_0:.*]] = aie.lock(%[[TILE_1_1]], 0) {init = 2 : i32, sym_name = "of2_prod_lock_0"}
// CHECK-DAG:     %[[OF2_CONS_LOCK_0:.*]] = aie.lock(%[[TILE_1_1]], 1) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK-DAG:     %[[OF2_PROD_LOCK_1:.*]] = aie.lock(%[[TILE_1_1]], 2) {init = 2 : i32, sym_name = "of2_prod_lock_1"}
// CHECK-DAG:     %[[OF2_CONS_LOCK_1:.*]] = aie.lock(%[[TILE_1_1]], 3) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK-DAG:     %[[OF1_BUFF_0:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of1_buff_0"} : memref<128xi32>
// CHECK-DAG:     %[[OF1_BUFF_1:.*]] = aie.buffer(%[[TILE_3_3]]) {sym_name = "of1_buff_1"} : memref<128xi32>
// CHECK-DAG:     %[[OF1_PROD_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 0) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK-DAG:     %[[OF1_CONS_LOCK:.*]] = aie.lock(%[[TILE_3_3]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK-DAG:     %[[OF0_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_0"} : memref<128xi32>
// CHECK-DAG:     %[[OF0_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "of0_buff_1"} : memref<128xi32>
// CHECK-DAG:     %[[OF0_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i32, sym_name = "of0_prod_lock_0"}
// CHECK-DAG:     %[[OF0_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK-DAG:     aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_1_1]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[TILE_3_3]], DMA : 0, %[[TILE_1_1]], DMA : 1)
// CHECK-DAG:     aie.flow(%[[TILE_1_1]], DMA : 0, %[[TILE_2_3]], DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%[[TILE_1_2]]) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[OF0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF0_BUFF_0]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[OF0_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[OF0_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF0_BUFF_1]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[OF0_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%[[TILE_1_1]]) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<256xi32>, 0, 128, [<size = 3, stride = 4>])
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<256xi32>, 0, 128, [<size = 3, stride = 4>])
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<256xi32>, 128, 128, [<size = 2, stride = 2>])
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<256xi32>, 128, 128, [<size = 2, stride = 2>])
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:
// CHECK:       %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb11)
// CHECK:     ^bb7:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<256xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb8
// CHECK:     ^bb8:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_0]] : memref<256xi32>, 128, 128)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb9:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<256xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_0]], Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:
// CHECK:       aie.use_lock(%[[OF2_CONS_LOCK_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_BUFF_1]] : memref<256xi32>, 128, 128)
// CHECK:       aie.use_lock(%[[OF2_PROD_LOCK_1]], Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb11:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%[[TILE_3_3]]) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[OF1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF1_BUFF_0]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[OF1_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[OF1_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF1_BUFF_1]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[OF1_PROD_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%[[TILE_2_3]]) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[OF2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_CONS_BUFF_0]] : memref<256xi32>, 0, 256)
// CHECK:       aie.use_lock(%[[OF2_CONS_CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[OF2_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[OF2_CONS_BUFF_1]] : memref<256xi32>, 0, 256)
// CHECK:       aie.use_lock(%[[OF2_CONS_CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile11 dimensionsFromStream [<size = 3, stride = 4>]}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of1 (%tile33, {%tile11 dimensionsFromStream [<size = 2, stride = 2>]}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of2 (%tile11, {%tile23}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo.link [ @of0, @of1 ] -> [ @of2 ] ([0, 128][])
 }
}
