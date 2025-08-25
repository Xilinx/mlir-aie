//===- packet_id_test.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos" %s | FileCheck %s

// CHECK: module @packet_id {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[VAL_0:.*]] = aie.tile(0, 0)
// CHECK:     %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK:     %[[VAL_2:.*]] = aie.tile(1, 2)
// CHECK:     %[[VAL_4:.*]] = aie.tile(3, 3)
// CHECK:     %[[VAL_5:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:     %[[VAL_6:.*]] = aie.buffer(%[[VAL_4]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:     %[[VAL_7:.*]] = aie.lock(%[[VAL_4]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:     %[[VAL_8:.*]] = aie.lock(%[[VAL_4]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:     %[[VAL_9:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:     %[[VAL_10:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:     %[[VAL_11:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:     %[[VAL_12:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     aie.packet_flow(2) {
// CHECK:        aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK:        aie.packet_dest<%[[VAL_4]], DMA : 0>
// CHECK:     }
// CHECK:     aie.packet_flow(1) {
// CHECK:        aie.packet_source<%[[VAL_1]], Trace : 0>
// CHECK:        aie.packet_dest<%[[VAL_0]], DMA : 1>
// CHECK:     } {keep_pkt_header = true}
// CHECK:     %[[VAL_13:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:       aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd_packet(0, 2)
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd_packet(0, 2)
// CHECK:       aie.dma_bd(%[[VAL_10]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[VAL_14:.*]] = aie.mem(%[[VAL_4]]) {
// CHECK:       aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_6]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @packet_id {
 aie.device(xcve2302) {
    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.packet_flow(1) {
      aie.packet_source<%tile02, Trace : 0>
      aie.packet_dest<%tile00, DMA : 1>
    } {keep_pkt_header = true}
 }
}
