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

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           memref.global "public" @link4_cons : memref<12xi32>
// CHECK:           memref.global "public" @link4 : memref<12xi32>
// CHECK:           memref.global "public" @link3_cons : memref<20xi32>
// CHECK:           memref.global "public" @link3 : memref<20xi32>
// CHECK:           memref.global "public" @link2_cons : memref<4x4xi32>
// CHECK:           memref.global "public" @link2 : memref<4x4xi32>
// CHECK:           memref.global "public" @link1_cons : memref<48xi32>
// CHECK:           memref.global "public" @link1 : memref<48xi32>
// CHECK:           %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[VAL_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK:           %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "link4_cons_buff_0"} : memref<12xi32>
// CHECK:           %[[VAL_6:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "link4_cons_buff_1"} : memref<12xi32>
// CHECK:           %[[VAL_7:.*]] = aie.lock(%[[VAL_4]], 0) {init = 2 : i32, sym_name = "link4_cons_prod_lock"}
// CHECK:           %[[VAL_8:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "link4_cons_cons_lock"}
// CHECK:           %[[VAL_9:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link3_cons_buff_0"} : memref<20xi32>
// CHECK:           %[[VAL_10:.*]] = aie.buffer(%[[VAL_3]]) {sym_name = "link3_cons_buff_1"} : memref<20xi32>
// CHECK:           %[[VAL_11:.*]] = aie.lock(%[[VAL_3]], 0) {init = 2 : i32, sym_name = "link3_cons_prod_lock"}
// CHECK:           %[[VAL_12:.*]] = aie.lock(%[[VAL_3]], 1) {init = 0 : i32, sym_name = "link3_cons_cons_lock"}
// CHECK:           %[[VAL_13:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "link2_cons_buff_0"} : memref<4x4xi32>
// CHECK:           %[[VAL_14:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "link2_cons_buff_1"} : memref<4x4xi32>
// CHECK:           %[[VAL_15:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "link2_cons_prod_lock"}
// CHECK:           %[[VAL_16:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "link2_cons_cons_lock"}
// CHECK:           %[[VAL_17:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_0"} : memref<48xi32>
// CHECK:           %[[VAL_18:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "link1_cons_buff_1"} : memref<48xi32>
// CHECK:           %[[VAL_19:.*]] = aie.lock(%[[VAL_1]], 0) {init = 6 : i32, sym_name = "link1_cons_prod_lock"}
// CHECK:           %[[VAL_20:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "link1_cons_cons_lock"}
// CHECK:           %[[VAL_21:.*]] = aie.lock(%[[VAL_0]], 0) {init = 1 : i32, sym_name = "link1_prod_lock"}
// CHECK:           %[[VAL_22:.*]] = aie.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "link1_cons_lock"}
// CHECK:           aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 1, %[[VAL_3]], DMA : 0)
// CHECK:           aie.flow(%[[VAL_1]], DMA : 2, %[[VAL_4]], DMA : 0)
// CHECK:           %[[VAL_23:.*]] = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<48xi32>
// CHECK:           aie.shim_dma_allocation @link1(MM2S, 0, 2)
// CHECK:           %[[VAL_24:.*]] = aie.shim_dma(%[[VAL_0]]) {
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:             aie.use_lock(%[[VAL_22]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_23]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_21]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = aie.memtile_dma(%[[VAL_1]]) {
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 3)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 3)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_19]], AcquireGreaterEqual, 3)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<48xi32>) {len = 48 : i32}
// CHECK:             aie.use_lock(%[[VAL_20]], Release, 3)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(MM2S, 0, ^bb4, ^bb6)
// CHECK:           ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<48xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:  // pred: ^bb4
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<48xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:  // pred: ^bb3
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 1, ^bb7, ^bb9)
// CHECK:           ^bb7:  // 2 preds: ^bb6, ^bb8
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<48xi32>) {len = 20 : i32, offset = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:  // pred: ^bb7
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<48xi32>) {len = 20 : i32, offset = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:  // pred: ^bb6
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(MM2S, 2, ^bb10, ^bb12)
// CHECK:           ^bb10:  // 2 preds: ^bb9, ^bb11
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_17]] : memref<48xi32>) {len = 12 : i32, offset = 36 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb11:  // pred: ^bb10
// CHECK:             aie.use_lock(%[[VAL_20]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_18]] : memref<48xi32>) {len = 12 : i32, offset = 36 : i32}
// CHECK:             aie.use_lock(%[[VAL_19]], Release, 1)
// CHECK:             aie.next_bd ^bb10
// CHECK:           ^bb12:  // pred: ^bb9
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_13]] : memref<4x4xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_15]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_14]] : memref<4x4xi32>) {len = 16 : i32}
// CHECK:             aie.use_lock(%[[VAL_16]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_33:.*]] = aie.mem(%[[VAL_3]]) {
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_9]] : memref<20xi32>) {len = 20 : i32}
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_10]] : memref<20xi32>) {len = 20 : i32}
// CHECK:             aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[VAL_35:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_5]] : memref<12xi32>) {len = 12 : i32}
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[VAL_6]] : memref<12xi32>) {len = 12 : i32}
// CHECK:             aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:  // pred: ^bb0
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @link_distribute {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile21 = aie.tile(2, 1)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %tile33 = aie.tile(3, 3)

        aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
        aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
        aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
        aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

        %ext_buffer_in  = aie.external_buffer {sym_name = "ext_buffer_in"}: memref<48xi32>
        aie.objectfifo.register_external_buffers @link1 (%tile20, {%ext_buffer_in}) : (memref<48xi32>)

        aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ()
    }
}
