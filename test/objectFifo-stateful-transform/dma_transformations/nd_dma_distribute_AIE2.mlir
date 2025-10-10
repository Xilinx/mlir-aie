//===- nd_dma_distribute_AIE2.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s

// Currently not supported: link should also support lengths in addition to offsets, currently length cannot be correctly inferred.

// CHECK: module @ndDMAObjFifoAIE2 {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[tile_1_0:.*]] = aie.tile(1, 0)
// CHECK:     %[[tile_1_1:.*]] = aie.tile(1, 1)
// CHECK:     %[[tile_2_2:.*]] = aie.tile(2, 2)
// CHECK:     %[[tile_2_3:.*]] = aie.tile(2, 3)
// CHECK:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "of2_cons_buff_0"} : memref<128xi32> 
// CHECK:     %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "of2_cons_buff_1"} : memref<128xi32> 
// CHECK:     %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_2_3, 0) {init = 2 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:     %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_2_3, 1) {init = 0 : i32, sym_name = "of2_cons_cons_lock_0"}
// CHECK:     %[[VAL_4:.*]] = aie.buffer(%{{.*}}tile_2_2) {sym_name = "of1_cons_buff_0"} : memref<128xi32> 
// CHECK:     %[[VAL_5:.*]] = aie.buffer(%{{.*}}tile_2_2) {sym_name = "of1_cons_buff_1"} : memref<128xi32> 
// CHECK:     %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_2_2, 0) {init = 2 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK:     %[[VAL_7:.*]] = aie.lock(%{{.*}}tile_2_2, 1) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// CHECK:     %[[VAL_8:.*]] = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of0_cons_buff_0"} : memref<256xi32> 
// CHECK:     %[[VAL_9:.*]] = aie.buffer(%{{.*}}tile_1_1) {sym_name = "of0_cons_buff_1"} : memref<256xi32> 
// CHECK:     %[[VAL_10:.*]] = aie.lock(%{{.*}}tile_1_1, 0) {init = 2 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:     %[[VAL_11:.*]] = aie.lock(%{{.*}}tile_1_1, 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:     %[[VAL_12:.*]] = aie.lock(%{{.*}}tile_1_1, 2) {init = 2 : i32, sym_name = "of0_cons_prod_lock_1"}
// CHECK:     %[[VAL_13:.*]] = aie.lock(%{{.*}}tile_1_1, 3) {init = 0 : i32, sym_name = "of0_cons_cons_lock_1"}
// CHECK:     %[[VAL_14:.*]] = aie.lock(%{{.*}}tile_1_0, 0) {init = 1 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[VAL_15:.*]] = aie.lock(%{{.*}}tile_1_0, 1) {init = 0 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:     aie.flow(%[[tile_1_0:.*]], DMA : 0, %[[tile_1_1:.*]], DMA : 0)
// CHECK:     aie.flow(%[[tile_1_1:.*]], DMA : 0, %[[tile_2_2:.*]], DMA : 0)
// CHECK:     aie.flow(%[[tile_1_1:.*]], DMA : 1, %[[tile_2_3:.*]], DMA : 0)
// CHECK:     aie.shim_dma_allocation @of0_shim_alloc(MM2S, 0, 1)
// CHECK:     %memtile_dma_1_1 = aie.memtile_dma(%{{.*}}tile_1_1) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb4
// CHECK:       aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<256xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<256xi32>, 128, 128)
// CHECK:       aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_10]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<256xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[VAL_11]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[VAL_12]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<256xi32>, 128, 128)
// CHECK:       aie.use_lock(%[[VAL_13]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb5:  // pred: ^bb0
// CHECK:       %1 = aie.dma_start(MM2S, 0, ^bb6, ^bb8)
// CHECK:     ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<256xi32>, 0, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:       aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:       aie.next_bd ^bb7
// CHECK:     ^bb7:  // pred: ^bb6
// CHECK:       aie.use_lock(%[[VAL_11]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<256xi32>, 0, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:       aie.use_lock(%[[VAL_10]], Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb8:  // pred: ^bb5
// CHECK:       %2 = aie.dma_start(MM2S, 1, ^bb9, ^bb11)
// CHECK:     ^bb9:  // 2 preds: ^bb8, ^bb10
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_8]] : memref<256xi32>, 512, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:       aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:       aie.next_bd ^bb10
// CHECK:     ^bb10:  // pred: ^bb9
// CHECK:       aie.use_lock(%[[VAL_13]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_9]] : memref<256xi32>, 512, 128, [<size = 4, stride = 64>, <size = 2, stride = 4>, <size = 8, stride = 8>, <size = 4, stride = 1>])
// CHECK:       aie.use_lock(%[[VAL_12]], Release, 1)
// CHECK:       aie.next_bd ^bb9
// CHECK:     ^bb11:  // pred: ^bb8
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_2 = aie.mem(%[[tile_2_2:.*]]) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_6]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[VAL_7]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%[[tile_2_3:.*]]) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_0]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_1]] : memref<128xi32>, 0, 128)
// CHECK:       aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @ndDMAObjFifoAIE2 {
 aie.device(xcve2302) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile10, {%tile11},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile11 dimensionsToStream [<size = 4, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile22}, 2 : i32) : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo @of2 (%tile11 dimensionsToStream [<size = 4, stride = 64>,
                                           <size = 2, stride = 4>,
                                           <size = 8, stride = 8>,
                                           <size = 4, stride = 1>],
                        {%tile23}, 2 : i32) : !aie.objectfifo<memref<128xi32>>
   aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ([][0, 512])
 }
}
