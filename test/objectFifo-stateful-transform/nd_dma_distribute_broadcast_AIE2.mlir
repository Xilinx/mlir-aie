//===- nd_dma_distribute_broadcast_AIE2_bad.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK-DAG:     memref.global "public" @[[OF2_0_CONS:.+]] : memref<128xi32>
// CHECK-DAG:     memref.global "public" @[[OF2_1_CONS:.+]] : memref<128xi32>
// CHECK-DAG:     memref.global "public" @[[OF2:.+]] : memref<128xi32>
// CHECK-DAG:     memref.global "public" @[[OF1_0_CONS:.+]] : memref<128xi32>
// CHECK-DAG:     memref.global "public" @[[OF1_1_CONS:.+]] : memref<128xi32>
// CHECK-DAG:     memref.global "public" @[[OF1:.+]] : memref<128xi32>
// CHECK-DAG:     memref.global "public" @[[OF0_CONS:.+]] : memref<256xi32>
// CHECK-DAG:     memref.global "public" @[[OF0:.+]] : memref<256xi32>
// CHECK-DAG:     %[[TILE_1_0:.+]] = aie.tile(1, 0)
// CHECK-DAG:     %[[TILE_1_1:.+]] = aie.tile(1, 1)
// CHECK-DAG:     %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:     %[[TILE_2_2:.+]] = aie.tile(2, 2)
// CHECK-DAG:     %[[TILE_1_3:.+]] = aie.tile(1, 3)
// CHECK-DAG:     %[[TILE_2_3:.+]] = aie.tile(2, 3)
// CHECK-DAG:     %[[OF2_0_CONS_BUFF_0:.+]] = aie.buffer(%[[TILE_1_3]])
// CHECK-DAG:     %[[OF2_0_CONS_BUFF_1:.+]] = aie.buffer(%[[TILE_1_3]])
// CHECK-DAG:     %[[OF2_0_CONS_PROD_LOCK:.+]] = aie.lock(%tile_1_3, 0) {init = 2 : i32
// CHECK-DAG:     %[[OF2_0_CONS_CONS_LOCK:.+]] = aie.lock(%tile_1_3, 1) {init = 0 : i32
// CHECK-DAG:     %[[OF2_1_CONS_BUFF_0:.+]] = aie.buffer(%[[TILE_2_3]])
// CHECK-DAG:     %[[OF2_1_CONS_BUFF_1:.+]] = aie.buffer(%[[TILE_2_3]])
// CHECK-DAG:     %[[OF2_1_CONS_PROD_LOCK:.+]] = aie.lock(%tile_2_3, 0) {init = 2 : i32
// CHECK-DAG:     %[[OF2_1_CONS_CONS_LOCK:.+]] = aie.lock(%tile_2_3, 1) {init = 0 : i32
// CHECK-DAG:     %[[OF1_0_CONS_BUFF_0:.+]] = aie.buffer(%[[TILE_1_2]])
// CHECK-DAG:     %[[OF1_0_CONS_BUFF_1:.+]] = aie.buffer(%[[TILE_1_2]])
// CHECK-DAG:     %[[OF1_0_CONS_PROD_LOCK:.+]] = aie.lock(%tile_1_2, 0) {init = 2 : i32
// CHECK-DAG:     %[[OF1_0_CONS_CONS_LOCK:.+]] = aie.lock(%tile_1_2, 1) {init = 0 : i32
// CHECK-DAG:     %[[OF1_1_CONS_BUFF_0:.+]] = aie.buffer(%[[TILE_2_2]])
// CHECK-DAG:     %[[OF1_1_CONS_BUFF_1:.+]] = aie.buffer(%[[TILE_2_2]])
// CHECK-DAG:     %[[OF1_1_CONS_PROD_LOCK:.+]] = aie.lock(%tile_2_2, 0) {init = 2 : i32
// CHECK-DAG:     %[[OF1_1_CONS_CONS_LOCK:.+]] = aie.lock(%tile_2_2, 1) {init = 0 : i32
// CHECK-DAG:     %[[OF0_CONS_BUFF_0:.+]] = aie.buffer(%[[TILE_1_1]])
// CHECK-DAG:     %[[OF0_CONS_BUFF_1:.+]] = aie.buffer(%[[TILE_1_1]])
// CHECK-DAG:     %[[OF0_CONS_PROD_LOCK:.+]] = aie.lock(%[[TILE_1_1]], 0) {init = 4 : i32
// CHECK-DAG:     %[[OF0_CONS_CONS_LOCK:.+]] = aie.lock(%[[TILE_1_1]], 1) {init = 0 : i32
// CHECK-DAG:     %[[OF0_PROD_LOCK:.+]] = aie.lock(%[[TILE_1_0]], 0) {init = 0 : i32
// CHECK-DAG:     %[[OF0_CONS_LOCK:.+]] = aie.lock(%[[TILE_1_0]], 1) {init = 0 : i32
// CHECK-DAG:     aie.flow(%[[TILE_1_0]], DMA : 0, %[[TILE_1_1]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[TILE_1_1]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[TILE_1_1]], DMA : 0, %[[TILE_1_2]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[TILE_1_1]], DMA : 1, %[[TILE_2_3]], DMA : 0)
// CHECK-DAG:     aie.flow(%[[TILE_1_1]], DMA : 1, %[[TILE_1_3]], DMA : 0)
// CHECK:         aie.shim_dma_allocation @[[OF0]](MM2S, 0, 1)
// CHECK:         %{{.+}} = aie.memtile_dma(%[[TILE_1_1]]) {
// CHECK:           %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], Release, 2)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], AcquireGreaterEqual, 2)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], Release, 2)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:              %[[VAL_1:.+]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>, dims = [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>, dims = [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]) {len = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:              %[[VAL_2:.+]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_0]] : memref<256xi32>, dims = [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]) {len = 128 : i32, offset = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[OF0_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OF0_CONS_BUFF_1]] : memref<256xi32>, dims = [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>]) {len = 128 : i32, offset = 128 : i32}
// CHECK:             aie.use_lock(%[[OF0_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:         }
// CHECK:         %{{.+}} = aie.mem(%[[TILE_1_2]]) {
// CHECK:           %{{.+}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[OF1_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF1_0_CONS_BUFF_0]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF1_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[OF1_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF1_0_CONS_BUFF_1]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF1_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %{{.+}} = aie.mem(%[[TILE_2_2]]) {
// CHECK:           %{{.+}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[OF1_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF1_1_CONS_BUFF_0]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF1_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[OF1_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF1_1_CONS_BUFF_1]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF1_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %{{.+}} = aie.mem(%[[TILE_1_3]]) {
// CHECK:           %{{.+}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[OF2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF2_0_CONS_BUFF_0]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[OF2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF2_0_CONS_BUFF_1]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }
// CHECK:         %{{.+}} = aie.mem(%[[TILE_2_3]]) {
// CHECK:           %{{.+}} = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[OF2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF2_1_CONS_BUFF_0]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb2
// CHECK:         ^bb2:
// CHECK:           aie.use_lock(%[[OF2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[OF2_1_CONS_BUFF_1]] : memref<128xi32>) {len = 128 : i32}
// CHECK:           aie.use_lock(%[[OF2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb3:
// CHECK:           aie.end
// CHECK:         }
aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile22 = aie.tile(2, 2)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)
    aie.objectfifo @of0 (%tile10, {%tile11}, 2 : i32) : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @of1 (%tile11 toStream [<size = 4, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile12, %tile22}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @of2 (%tile11 toStream [<size = 4, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile13, %tile23}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
   aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ()
}
