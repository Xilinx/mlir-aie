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

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           memref.global "public" @skip_connection_cons : memref<16xi32>
// CHECK:           memref.global "public" @skip_connection : memref<16xi32>
// CHECK:           memref.global "public" @link2_0_cons : memref<16xi32>
// CHECK:           memref.global "public" @link2_1_cons : memref<16xi32>
// CHECK:           memref.global "public" @link2 : memref<16xi32>
// CHECK:           memref.global "public" @link1_cons : memref<48xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "skip_connection_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "skip_connection_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_3]], 2) {init = 2 : i32, sym_name = "skip_connection_cons_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_3]], 3) {init = 0 : i32, sym_name = "skip_connection_cons_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "skip_connection_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "skip_connection_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_2]], 2) {init = 2 : i32, sym_name = "skip_connection_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_2]], 3) {init = 0 : i32, sym_name = "skip_connection_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "link2_0_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_13:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "link2_0_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_14:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "link2_0_cons_prod_lock"}
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "link2_0_cons_cons_lock"}
// CHECK:           %[[VAL_16:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link2_1_cons_buff_0"} : memref<16xi32>
// CHECK:           %[[VAL_17:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link2_1_cons_buff_1"} : memref<16xi32>
// CHECK:           %[[VAL_18:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link2_1_cons_buff_2"} : memref<16xi32>
// CHECK:           %[[VAL_19:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 3 : i32, sym_name = "link2_1_cons_prod_lock"}
// CHECK:           %[[VAL_20:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "link2_1_cons_cons_lock"}
// CHECK:           %[[VAL_21:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_22:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_23:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 2 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:           %[[VAL_24:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:           %[[VAL_25:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[VAL_26:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_3]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_3]], DMA : 1)
// CHECK:           AIE.shimDMAAllocation @link1(MM2S, 0, 2)
// CHECK:           %[[VAL_27:.*]] = AIE.memTileDMA(%[[VAL_1]]) {
// CHECK:             %[[VAL_28:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_23]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_21]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_24]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_23]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_22]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_24]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_29:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_24]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_21]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_23]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_24]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_22]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_23]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_30:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_31:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_12]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_14]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_13]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_15]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_32:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_10]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_34:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_16]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_20]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_17]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_20]], Release, 1)
// CHECK:             AIE.nextBd ^bb3
// CHECK:           ^bb3:  // pred: ^bb2
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_18]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_20]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb4:  // pred: ^bb0
// CHECK:             %[[VAL_35:.*]] = AIE.dmaStart(S2MM, 1, ^bb5, ^bb7)
// CHECK:           ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK:             AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_4]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb6
// CHECK:           ^bb6:  // pred: ^bb5
// CHECK:             AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<16xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb7:  // pred: ^bb4
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @link_broadcast {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)
        %tile33 = AIE.tile(3, 3)

        AIE.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<48xi32>>
        AIE.objectfifo @link2 (%tile21, {%tile22, %tile33}, [2, 2, 3]) : !AIE.objectfifo<memref<16xi32>>

        AIE.objectfifo @skip_connection (%tile22, {%tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>

        AIE.objectfifo.link [@link1] -> [@link2] ()
    }
}
