//===- base_test_AIE1.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: July 26th 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @elementGenerationAIE1 {
// CHECK:   AIE.device(xcvc1902) {
// CHECK:     %[[VAL_0:.*]] = AIE.tile(1, 2)
// CHECK:     %[[VAL_1:.*]] = AIE.tile(1, 3)
// CHECK:     %[[VAL_2:.*]] = AIE.tile(3, 3)
// CHECK:     %[[VAL_3:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:     %[[VAL_4:.*]] = AIE.buffer(%[[VAL_2]]) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:     %[[VAL_5:.*]] = AIE.lock(%[[VAL_2]], 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     %[[VAL_6:.*]] = AIE.lock(%[[VAL_2]], 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:     %[[VAL_7:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:     %[[VAL_8:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:     %[[VAL_9:.*]] = AIE.lock(%[[VAL_0]], 4) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:     %[[VAL_10:.*]] = AIE.lock(%[[VAL_0]], 5) {init = 0 : i32, sym_name = "of1_lock_1"}
// CHECK:     %[[VAL_11:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:     %[[VAL_12:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK:     %[[VAL_13:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of0_buff_2"} : memref<16xi32>
// CHECK:     %[[VAL_14:.*]] = AIE.buffer(%[[VAL_0]]) {sym_name = "of0_buff_3"} : memref<16xi32>
// CHECK:     %[[VAL_15:.*]] = AIE.lock(%[[VAL_0]], 0) {init = 0 : i32, sym_name = "of0_lock_0"}
// CHECK:     %[[VAL_16:.*]] = AIE.lock(%[[VAL_0]], 1) {init = 0 : i32, sym_name = "of0_lock_1"}
// CHECK:     %[[VAL_17:.*]] = AIE.lock(%[[VAL_0]], 2) {init = 0 : i32, sym_name = "of0_lock_2"}
// CHECK:     %[[VAL_18:.*]] = AIE.lock(%[[VAL_0]], 3) {init = 0 : i32, sym_name = "of0_lock_3"}
// CHECK:     AIE.flow(%[[VAL_0]], DMA : 0, %[[VAL_2]], DMA : 0)
// CHECK:     %[[VAL_19:.*]] = AIE.mem(%[[VAL_0]]) {
// CHECK:       AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%[[VAL_9]], Acquire, 1)
// CHECK:       AIE.dmaBd(<%[[VAL_7]] : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%[[VAL_9]], Release, 0)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[VAL_10]], Acquire, 1)
// CHECK:       AIE.dmaBd(<%[[VAL_8]] : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%[[VAL_10]], Release, 0)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %[[VAL_20:.*]] = AIE.mem(%[[VAL_2]]) {
// CHECK:       AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%[[VAL_5]], Acquire, 0)
// CHECK:       AIE.dmaBd(<%[[VAL_3]] : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%[[VAL_5]], Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%[[VAL_6]], Acquire, 0)
// CHECK:       AIE.dmaBd(<%[[VAL_4]] : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%[[VAL_6]], Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @elementGenerationAIE1 {
   AIE.device(xcvc1902) {
      %tile12 = AIE.tile(1, 2)
      %tile13 = AIE.tile(1, 3)
      %tile33 = AIE.tile(3, 3)

      // In the shared memory case, the number of elements does not change.
      AIE.objectfifo @of0 (%tile12, {%tile13}, 4 : i32) : !AIE.objectfifo<memref<16xi32>>

      // In the non-adjacent memory case, the number of elements depends on the max amount acquired by
      // the processes running on each core (here nothing is specified so it cannot be derived).
      AIE.objectfifo @of1 (%tile12, {%tile33}, 2 : i32) : !AIE.objectfifo<memref<16xi32>>
   }
}
