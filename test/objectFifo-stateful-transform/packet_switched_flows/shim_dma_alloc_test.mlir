//===- shim_dma_alloc_test.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform="packet-sw-objFifos" %s | FileCheck %s

// CHECK: module @shim_dma_alloc {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:     %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:     %[[VAL_3:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:     %[[VAL_4:.*]] = aie.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:     %[[VAL_5:.*]] = aie.lock(%[[VAL_2]], 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:     %[[VAL_6:.*]] = aie.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:     aie.packet_flow(0) {
// CHECK:        aie.packet_source<%[[VAL_0]], DMA : 0>
// CHECK:        aie.packet_dest<%[[VAL_2]], DMA : 0>
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of1(MM2S, 0, 1, <pkt_type = 0, pkt_id = 0>)
// CHECK:     %[[VAL_12:.*]] = aie.mem(%[[VAL_2]]) {
// CHECK:       aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_3]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @shim_dma_alloc {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of1 (%tile10, {%tile33}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
 }
}
