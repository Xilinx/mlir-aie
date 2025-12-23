//===- init_values_test.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @init {
// CHECK:   aie.device(xcve2302) {
// CHECK:     %{{.*}}tile_1_2 = aie.tile(1, 2)
// CHECK:     %{{.*}}tile_2_3 = aie.tile(2, 3)
// CHECK:     %[[VAL_0:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "of0_cons_buff_0"} : memref<2x2xi32> 
// CHECK:     %[[VAL_1:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "of0_cons_buff_1"} : memref<2x2xi32> 
// CHECK:     %[[VAL_2:.*]] = aie.buffer(%{{.*}}tile_2_3) {sym_name = "of0_cons_buff_2"} : memref<2x2xi32> 
// CHECK:     %[[VAL_3:.*]] = aie.lock(%{{.*}}tile_2_3, 0) {init = 3 : i32, sym_name = "of0_cons_prod_lock_0"}
// CHECK:     %[[VAL_4:.*]] = aie.lock(%{{.*}}tile_2_3, 1) {init = 0 : i32, sym_name = "of0_cons_cons_lock_0"}
// CHECK:     %[[VAL_5:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_0"} : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
// CHECK:     %[[VAL_6:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_1"} : memref<2x2xi32> = dense<{{\[}}[4, 5], [6, 7]]>
// CHECK:     %[[VAL_7:.*]] = aie.buffer(%{{.*}}tile_1_2) {sym_name = "of0_buff_2"} : memref<2x2xi32> = dense<{{\[}}[8, 9], [10, 11]]>
// CHECK:     %[[VAL_8:.*]] = aie.lock(%{{.*}}tile_1_2, 0) {init = 0 : i32, sym_name = "of0_prod_lock_0"}
// CHECK:     %[[VAL_9:.*]] = aie.lock(%{{.*}}tile_1_2, 1) {init = 3 : i32, sym_name = "of0_cons_lock_0"}
// CHECK:     aie.flow(%{{.*}}tile_1_2, DMA : 0, %{{.*}}tile_2_3, DMA : 0)
// CHECK:     %mem_1_2 = aie.mem(%{{.*}}tile_1_2) {
// CHECK:       %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_5]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_6]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_9]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_7]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_8]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %mem_2_3 = aie.mem(%{{.*}}tile_2_3) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_0]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_1]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       aie.use_lock(%[[VAL_3]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[VAL_2]] : memref<2x2xi32>, 0, 4)
// CHECK:       aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @init {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile12, {%tile23}, 3 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[0, 1], [2, 3]]> : memref<2x2xi32>, 
                                                                                            dense<[[4, 5], [6, 7]]> : memref<2x2xi32>,
                                                                                            dense<[[8, 9], [10, 11]]> : memref<2x2xi32>]
 }
}
