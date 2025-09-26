//===- link_join_repeat_count_test.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @memtileRepeat {
// CHECK:   aie.device(npu1) {
// CHECK:     memref.global "public" @of2_cons : memref<32xi32>
// CHECK:     memref.global "public" @of2 : memref<32xi32>
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     memref.global "public" @of0_cons : memref<16xi32>
// CHECK:     memref.global "public" @of0 : memref<16xi32>
// CHECK:     %{{.*}}tile_1_0 = aie.tile(1, 0)
// CHECK:     %{{.*}}tile_1_1 = aie.tile(1, 1)
// CHECK:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:     %{{.*}}tile_3_3 = aie.tile(3, 3)
// CHECK:     %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of2_buff_0"} : memref<32xi32> 
// CHECK:     %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_1_1, 0) {init = 1 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:     %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_1_1, 1) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
// CHECK:     %[[VAL_5:.*]] = aie.lock(%{{.*}}tile_1_1, 2) {init = 1 : i32, sym_name = "of2_prod_lock_1"}
// CHECK:     %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_1_1, 3) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
// CHECK:     %[[VAL_7:.*]] = aie.buffer(%{{.*}}tile_3_3) {sym_name = "of1_buff_0"} : memref<16xi32> 
// CHECK:     %[[VAL_8:.*]] = aie.lock(%{{.*}}tile_3_3, 0) {init = 3 : i32, sym_name = "of1_prod_lock_0"}
// CHECK:     %[[VAL_9:.*]] = aie.lock(%{{.*}}tile_3_3, 1) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     %[[VAL_10:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_0"} : memref<16xi32> 
// CHECK:     %[[VAL_11:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 3 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[VAL_12:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_1, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}tile_1_1, DMA : 1)
// CHECK:     aie.flow(%{{.*}}tile_1_1, DMA : 0, %{{.*}}tile_1_0, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_10]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_10]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_10]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_2]] : memref<32xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       aie.use_lock(%[[VAL_5]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_2]] : memref<32xi32>, 16, 16)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       %2 = aie.dma_start(MM2S, 0, ^bb5, ^bb7)
// CHECK:     ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK:       aie.use_lock(%[[VAL_4]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_2]] : memref<32xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_2]] : memref<32xi32>, 16, 16)
// CHECK:       aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb7:  // pred: ^bb4
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_3_3 = aie.mem(%{{.*}}tile_3_3) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_7]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_7]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_7]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @of2(S2MM, 0, 1)
// CHECK:   }
// CHECK: }

module @memtileRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile12, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of1 (%tile33, {%tile11}, 1 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile11, {%tile10}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@of0, @of1] -> [@of2] ([0, 16] [])
 }
}
