//===- link_test_distribute.mlir ------------------------------------------------*- MLIR -*-===//
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
// CHECK:           memref.global "public" @link4_cons : memref<12xi32>
// CHECK:           memref.global "public" @link4 : memref<12xi32>
// CHECK:           memref.global "public" @link3_cons : memref<20xi32>
// CHECK:           memref.global "public" @link3 : memref<20xi32>
// CHECK:           memref.global "public" @link2_cons : memref<4x4xi32>
// CHECK:           memref.global "public" @link2 : memref<4x4xi32>
// CHECK:           memref.global "public" @link1_cons : memref<48xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "link4_cons_buff_0"} : memref<12xi32>
// CHECK:           %[[VAL_6:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "link4_cons_buff_1"} : memref<12xi32>
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_4]], 0) {init = 2 : i32, sym_name = "link4_cons_prod_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "link4_cons_cons_lock"}
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link3_cons_buff_0"} : memref<20xi32>
// CHECK:           %[[VAL_10:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link3_cons_buff_1"} : memref<20xi32>
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "link3_cons_prod_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "link3_cons_cons_lock"}
// CHECK:           %[[VAL_13:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "link2_cons_buff_0"} : memref<4x4xi32>
// CHECK:           %[[VAL_14:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "link2_cons_buff_1"} : memref<4x4xi32>
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "link2_cons_prod_lock"}
// CHECK:           %[[VAL_16:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "link2_cons_cons_lock"}
// CHECK:           %[[VAL_17:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_18:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_19:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 6 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:           %[[VAL_20:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:           %[[VAL_21:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[VAL_22:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 2, %[[VAL_4]], DMA : 0)
// CHECK:           %[[VAL_23:.*]] = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<48xi32>
// CHECK:           AIE.shimDMAAllocation @link1(MM2S, 0, 2)
// CHECK:           %[[VAL_24:.*]] = AIE.shimDMA(%[[VAL_0]]) {
// CHECK:             %[[VAL_25:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             AIE.useLock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_23]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_21]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = AIE.memTileDMA(%[[VAL_1]]) {
// CHECK:             %[[VAL_27:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 3)
// CHECK:             AIE.dmaBd(<%[[VAL_17]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_20]], Release, 3)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 3)
// CHECK:             AIE.dmaBd(<%[[VAL_18]] : memref<48xi32>, 0, 48>, 0)
// CHECK:             AIE.useLock(%[[VAL_20]], Release, 3)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_28:.*]] = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_17]] : memref<48xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_18]] : memref<48xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             %[[VAL_29:.*]] = AIE.dmaStart(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:             AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_17]] : memref<48xi32>, 64, 20>, 0)
// CHECK:             AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:             AIE.nextBd ^bb8
// CHECK:           ^bb8:  // pred: ^bb7
// CHECK:             AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_18]] : memref<48xi32>, 64, 20>, 0)
// CHECK:             AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:             AIE.nextBd ^bb7
// CHECK:           ^bb9:  // pred: ^bb6
// CHECK:             %[[VAL_30:.*]] = AIE.dmaStart(MM2S, 2, ^bb10, ^bb12)
// CHECK:           ^bb10:  // 2 preds: ^bb9, ^bb11
// CHECK:             AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_17]] : memref<48xi32>, 144, 12>, 0)
// CHECK:             AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:             AIE.nextBd ^bb11
// CHECK:           ^bb11:  // pred: ^bb10
// CHECK:             AIE.useLock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_18]] : memref<48xi32>, 144, 12>, 0)
// CHECK:             AIE.useLock(%[[VAL_19]], Release, 1)
// CHECK:             AIE.nextBd ^bb10
// CHECK:           ^bb12:  // pred: ^bb9
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_32:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_13]] : memref<4x4xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_14]] : memref<4x4xi32>, 0, 16>, 0)
// CHECK:             AIE.useLock(%[[VAL_16]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = AIE.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_34:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<20xi32>, 0, 20>, 0)
// CHECK:             AIE.useLock(%[[VAL_12]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_10]] : memref<20xi32>, 0, 20>, 0)
// CHECK:             AIE.useLock(%[[VAL_12]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_36:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_5]] : memref<12xi32>, 0, 12>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_6]] : memref<12xi32>, 0, 12>, 0)
// CHECK:             AIE.useLock(%[[VAL_8]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @link_distribute {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %tile33 = AIE.tile(3, 3)

        AIE.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<48xi32>>
        AIE.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !AIE.objectfifo<memref<4x4xi32>>
        AIE.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !AIE.objectfifo<memref<20xi32>>
        AIE.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !AIE.objectfifo<memref<12xi32>>

        %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<48xi32>
        AIE.objectfifo.register_external_buffers @link1 (%tile20, {%ext_buffer_in}) : (memref<48xi32>)

        AIE.objectfifo.link [@link1] -> [@link2, @link3, @link4] ()
    }
}
