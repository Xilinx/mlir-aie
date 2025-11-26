//===- producer_stream_AIE2.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @producer_stream_AIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[VAL_0:.*]] = aie.tile(1, 2)
// CHECK:     %[[VAL_1:.*]] = aie.tile(1, 3)
// CHECK:     %[[VAL_2:.*]] = aie.tile(3, 3)
// CHECK:     %of_producer_stream_cons_buff_0 = aie.buffer(%tile_1_3) {sym_name = "of_producer_stream_cons_buff_0"} : memref<16xi32> 
// CHECK:     %of_producer_stream_cons_buff_1 = aie.buffer(%tile_1_3) {sym_name = "of_producer_stream_cons_buff_1"} : memref<16xi32> 
// CHECK:     %of_producer_stream_cons_buff_2 = aie.buffer(%tile_1_3) {sym_name = "of_producer_stream_cons_buff_2"} : memref<16xi32> 
// CHECK:     %of_producer_stream_cons_prod_lock_0 = aie.lock(%tile_1_3, 0) {init = 3 : i32, sym_name = "of_producer_stream_cons_prod_lock_0"}
// CHECK:     %of_producer_stream_cons_cons_lock_0 = aie.lock(%tile_1_3, 1) {init = 0 : i32, sym_name = "of_producer_stream_cons_cons_lock_0"}
// CHECK:     aie.flow(%tile_1_2, Core : 0, %tile_1_3, DMA : 0)
// CHECK:     %mem_1_3 = aie.mem(%tile_1_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%of_producer_stream_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_producer_stream_cons_buff_0 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_producer_stream_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%of_producer_stream_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_producer_stream_cons_buff_1 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_producer_stream_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%of_producer_stream_cons_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%of_producer_stream_cons_buff_2 : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%of_producer_stream_cons_cons_lock_0, Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @producer_stream_AIE2 {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of_producer_stream (%tile12, {%tile13}, 3 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
