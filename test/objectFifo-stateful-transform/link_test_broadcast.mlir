//===- link_test_broadcast.mlir ------------------------------------------------*- MLIR -*-===//
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

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @skip_connection_cons : memref<16xi32>
// CHECK:           memref.global "public" @skip_connection : memref<16xi32>
// CHECK:           memref.global "public" @link2_0_cons : memref<16xi32>
// CHECK:           memref.global "public" @link2_1_cons : memref<16xi32>
// CHECK:           memref.global "public" @link2 : memref<16xi32>
// CHECK:           memref.global "public" @link1_cons : memref<48xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "skip_connection_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "skip_connection_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = aie.lock(%[[VAL_3]], 2) {init = 2 : i32, sym_name = "skip_connection_cons_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_3]], 3) {init = 0 : i32, sym_name = "skip_connection_cons_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "skip_connection_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "skip_connection_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = aie.lock(%[[VAL_2]], 2) {init = 2 : i32, sym_name = "skip_connection_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_2]], 3) {init = 0 : i32, sym_name = "skip_connection_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "link2_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "link2_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_14:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "link2_0_cons_prod_lock"}
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "link2_0_cons_cons_lock"}
// CHECK:           %[[VAL_16:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link2_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link2_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link2_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_3]], 0) {init = 3 : i32, sym_name = "link2_1_cons_prod_lock"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "link2_1_cons_cons_lock"}
// CHECK:           %[[VAL_21:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_22:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_23:.*]] = aie.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:           %[[VAL_24:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:           %[[VAL_25:.*]] = aie.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[VAL_26:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 1)
// CHECK:           aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:           %[[VAL_27:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_23]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_24]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_23]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_24]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(%[[VAL_24]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_21]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_23]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(%[[VAL_24]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_22]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_23]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_12]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_15]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_8]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_16]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(S2MM, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_4]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb6:  // pred: ^bb5
// CHECK:             aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<16xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb7:  // pred: ^bb4
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_broadcast {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22, %tile33}, [2, 2, 3]) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        aie.objectfifo.link [@link1] -> [@link2] ()
    }
}
