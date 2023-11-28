//===- link_test_join.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: June 30th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   AIE.device(xcve2302) {
// CHECK:           memref.global "public" @link5_cons : memref<512xi8>
// CHECK:           memref.global "public" @link5 : memref<512xi8>
// CHECK:           memref.global "public" @link4_cons : memref<128xi8>
// CHECK:           memref.global "public" @link4 : memref<128xi8>
// CHECK:           memref.global "public" @link3_cons : memref<128xi8>
// CHECK:           memref.global "public" @link3 : memref<128xi8>
// CHECK:           memref.global "public" @link2_cons : memref<128xi8>
// CHECK:           memref.global "public" @link2 : memref<128xi8>
// CHECK:           memref.global "public" @link1_cons : memref<128xi8>
// CHECK:           memref.global "public" @link1 : memref<128xi8>
// CHECK:           %[[VAL_0:.*]] = AIE.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = AIE.tile(1, 2)
// CHECK:           %[[VAL_3:.*]] = AIE.tile(2, 2)
// CHECK:           %[[VAL_4:.*]] = AIE.tile(2, 3)
// CHECK:           %[[VAL_5:.*]] = AIE.tile(3, 3)
// CHECK:           %[[VAL_6:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "link5_cons_prod_lock"}
// CHECK:           %[[VAL_7:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "link5_cons_cons_lock"}
// CHECK:           %[[VAL_8:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "link5_buff_0"} : memref<512xi8>
// CHECK:           %[[VAL_9:.*]] = AIE.buffer(%[[VAL_1]]) {sym_name = "link5_buff_1"} : memref<512xi8>
// CHECK:           %[[VAL_10:.*]] = AIE.lock(%[[VAL_1]], 0) {init = 8 : i32, sym_name = "link5_prod_lock"}
// CHECK:           %[[VAL_11:.*]] = AIE.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "link5_cons_lock"}
// CHECK:           %[[VAL_12:.*]] = AIE.buffer(%[[VAL_5]]) {sym_name = "link4_buff_0"} : memref<128xi8>
// CHECK:           %[[VAL_13:.*]] = AIE.buffer(%[[VAL_5]]) {sym_name = "link4_buff_1"} : memref<128xi8>
// CHECK:           %[[VAL_14:.*]] = AIE.lock(%[[VAL_5]], 0) {init = 2 : i32, sym_name = "link4_prod_lock"}
// CHECK:           %[[VAL_15:.*]] = AIE.lock(%[[VAL_5]], 1) {init = 0 : i32, sym_name = "link4_cons_lock"}
// CHECK:           %[[VAL_16:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "link3_buff_0"} : memref<128xi8>
// CHECK:           %[[VAL_17:.*]] = AIE.buffer(%[[VAL_4]]) {sym_name = "link3_buff_1"} : memref<128xi8>
// CHECK:           %[[VAL_18:.*]] = AIE.lock(%[[VAL_4]], 0) {init = 2 : i32, sym_name = "link3_prod_lock"}
// CHECK:           %[[VAL_19:.*]] = AIE.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "link3_cons_lock"}
// CHECK:           %[[VAL_20:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link2_buff_0"} : memref<128xi8>
// CHECK:           %[[VAL_21:.*]] = AIE.buffer(%[[VAL_3]]) {sym_name = "link2_buff_1"} : memref<128xi8>
// CHECK:           %[[VAL_22:.*]] = AIE.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "link2_prod_lock"}
// CHECK:           %[[VAL_23:.*]] = AIE.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "link2_cons_lock"}
// CHECK:           %[[VAL_24:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "link1_buff_0"} : memref<128xi8>
// CHECK:           %[[VAL_25:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "link1_buff_1"} : memref<128xi8>
// CHECK:           %[[VAL_26:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[VAL_27:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           AIE.flow(%[[VAL_2]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           AIE.flow(%[[VAL_3]], DMA : 0, %[[VAL_1]], DMA : 1)
// CHECK:           AIE.flow(%[[VAL_4]], DMA : 0, %[[VAL_1]], DMA : 2)
// CHECK:           AIE.flow(%[[VAL_5]], DMA : 0, %[[VAL_1]], DMA : 3)
// CHECK:           AIE.flow(%[[VAL_1]], DMA : 0, %[[VAL_0]], DMA : 0)
// CHECK:           %[[VAL_28:.*]] = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<512xi8>
// CHECK:           %[[VAL_29:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_30:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_27]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_24]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_26]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_27]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_25]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_26]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = AIE.memTileDMA(%[[VAL_1]]) {
// CHECK:             %[[VAL_32:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<512xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<512xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_33:.*]] = AIE.dmaStart(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<512xi8>, 128, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<512xi8>, 128, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             %[[VAL_34:.*]] = AIE.dmaStart(S2MM, 2, ^bb7, ^bb9)
// CHECK:           ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<512xi8>, 256, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb8
// CHECK:           ^bb8:  // pred: ^bb7
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<512xi8>, 256, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb7
// CHECK:           ^bb9:  // pred: ^bb6
// CHECK:             %[[VAL_35:.*]] = AIE.dmaStart(S2MM, 3, ^bb10, ^bb12)
// CHECK:           ^bb10:  // 2 preds: ^bb9, ^bb11
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<512xi8>, 384, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb11
// CHECK:           ^bb11:  // pred: ^bb10
// CHECK:             AIE.useLock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<512xi8>, 384, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_11]], Release, 1)
// CHECK:             AIE.nextBd ^bb10
// CHECK:           ^bb12:  // pred: ^bb9
// CHECK:             %[[VAL_36:.*]] = AIE.dmaStart(MM2S, 0, ^bb13, ^bb15)
// CHECK:           ^bb13:  // 2 preds: ^bb12, ^bb14
// CHECK:             AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 4)
// CHECK:             AIE.dmaBd(<%[[VAL_8]] : memref<512xi8>, 0, 512>, 0)
// CHECK:             AIE.useLock(%[[VAL_10]], Release, 4)
// CHECK:             AIE.nextBd ^bb14
// CHECK:           ^bb14:  // pred: ^bb13
// CHECK:             AIE.useLock(%[[VAL_11]], AcquireGreaterEqual, 4)
// CHECK:             AIE.dmaBd(<%[[VAL_9]] : memref<512xi8>, 0, 512>, 0)
// CHECK:             AIE.useLock(%[[VAL_10]], Release, 4)
// CHECK:             AIE.nextBd ^bb13
// CHECK:           ^bb15:  // pred: ^bb12
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_37:.*]] = AIE.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_38:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_23]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_20]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_22]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_23]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_21]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_22]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = AIE.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_40:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_16]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_18]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_19]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_17]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_18]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           %[[VAL_41:.*]] = AIE.mem(%[[VAL_5]]) {
// CHECK:             %[[VAL_42:.*]] = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             AIE.useLock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_12]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:             AIE.nextBd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             AIE.useLock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_13]] : memref<128xi8>, 0, 128>, 0)
// CHECK:             AIE.useLock(%[[VAL_14]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:           AIE.shimDMAAllocation @link5(S2MM, 0, 2)
// CHECK:           %[[VAL_43:.*]] = AIE.shimDMA(%[[VAL_0]]) {
// CHECK:             %[[VAL_44:.*]] = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             AIE.useLock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:             AIE.dmaBd(<%[[VAL_28]] : memref<512xi8>, 0, 512>, 0)
// CHECK:             AIE.useLock(%[[VAL_7]], Release, 1)
// CHECK:             AIE.nextBd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             AIE.end
// CHECK:           }
// CHECK:         }

module @link_join {
    AIE.device(xcve2302) {
        %tile20 = AIE.tile(2, 0)
        %tile21 = AIE.tile(2, 1)
        %tile12 = AIE.tile(1, 2)
        %tile22 = AIE.tile(2, 2)
        %tile23 = AIE.tile(2, 3)
        %tile33 = AIE.tile(3, 3)

        AIE.objectfifo @link1 (%tile12, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<128xi8>>
        AIE.objectfifo @link2 (%tile22, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<128xi8>>
        AIE.objectfifo @link3 (%tile23, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<128xi8>>
        AIE.objectfifo @link4 (%tile33, {%tile21}, 2 : i32) : !AIE.objectfifo<memref<128xi8>>
        AIE.objectfifo @link5 (%tile21, {%tile20}, 2 : i32) : !AIE.objectfifo<memref<512xi8>>

        %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<512xi8>
        AIE.objectfifo.register_external_buffers @link5 (%tile20, {%ext_buffer_in}) : (memref<512xi8>)

        AIE.objectfifo.link [@link1, @link2, @link3, @link4] -> [@link5] ()
    }
}
