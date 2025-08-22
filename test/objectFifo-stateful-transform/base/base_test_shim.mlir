//===- base_test_shim.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @elementGenerationShim {
// CHECK:   aie.device(xcvc1902) {
// CHECK:     memref.global "public" @of1_cons : memref<16xi32>
// CHECK:     memref.global "public" @of1 : memref<16xi32>
// CHECK:     %[[VAL_0:.*]] = aie.tile(1, 0)
// CHECK:     %[[VAL_1:.*]] = aie.tile(1, 2)
// CHECK:     %[[VAL_2:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32> 
// CHECK:     %[[VAL_3:.*]] = aie.buffer(%[[VAL_1]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32> 
// CHECK:     %[[VAL_4:.*]] = aie.lock(%[[VAL_1]], 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     %[[VAL_5:.*]] = aie.lock(%[[VAL_1]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:     aie.flow(%[[VAL_0]], DMA : 0, %[[VAL_1]], DMA : 0)
// CHECK:     aie.shim_dma_allocation @of1(MM2S, 0, 1)
// CHECK:     %[[VAL_6:.*]] = aie.mem(%[[VAL_1]]) {
// CHECK:       %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       aie.use_lock(%[[VAL_4]], Acquire, 0)
// CHECK:       aie.dma_bd(%[[VAL_2]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_4]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       aie.use_lock(%[[VAL_5]], Acquire, 0)
// CHECK:       aie.dma_bd(%[[VAL_3]] : memref<16xi32>, 0, 16)
// CHECK:       aie.use_lock(%[[VAL_5]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       aie.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @elementGenerationShim {
   aie.device(xcvc1902) {
      %tile10 = aie.tile(1, 0)
      %tile12 = aie.tile(1, 2)

      aie.objectfifo @of1 (%tile10, {%tile12}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
   }
}
