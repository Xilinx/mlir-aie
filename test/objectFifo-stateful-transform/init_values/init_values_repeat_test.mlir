//===- init_values_repeat_test.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @init_repeat {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:     %{{.*}}tile_1_3 = aie.tile(1, 3)
// CHECK:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of0_cons_buff_0"} : memref<2x2xi32> 
// CHECK:     %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_1_3) {sym_name = "of0_cons_buff_1"} : memref<2x2xi32> 
// CHECK:     %[[VAL_2:.*]] = aie.lock(%{{.*}}tile_1_3, 0) {init = 2 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:     %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_1_3, 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:     %[[VAL_4:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK:     %[[VAL_5:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[}}[4, 5], [6, 7]]>
// CHECK:     %[[VAL_6:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[VAL_7:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 6 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_1_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb7)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb6
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_4]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb4:  // pred: ^bb3
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb6
// CHECK:     ^bb6:  // pred: ^bb5
// CHECK:       aie.use_lock(%[[VAL_7]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_6]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb7:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_1_3 = aie.mem(%{{.*}}tile_1_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_0]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_1]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_3]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @init_repeat {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of0 (%tile12, {%tile13}, 2 : i32) {repeat_count = 3 : i32} : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>, 
                                                                                                                     dense<[[4, 5], [6, 7]]> : memref<2x2xi32>]
 }
}
