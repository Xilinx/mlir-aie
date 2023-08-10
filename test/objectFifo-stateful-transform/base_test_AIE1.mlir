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
// CHECK:     %0 = AIE.tile(1, 2)
// CHECK:     %1 = AIE.tile(1, 3)
// CHECK:     %2 = AIE.tile(3, 3)
// CHECK:     %3 = AIE.buffer(%0) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:     %4 = AIE.buffer(%0) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK:     %5 = AIE.buffer(%0) {sym_name = "of0_buff_2"} : memref<16xi32>
// CHECK:     %6 = AIE.buffer(%0) {sym_name = "of0_buff_3"} : memref<16xi32>
// CHECK:     %7 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "of0_lock_0"}
// CHECK:     %8 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of0_lock_1"}
// CHECK:     %9 = AIE.lock(%0, 2) {init = 0 : i32, sym_name = "of0_lock_2"}
// CHECK:     %10 = AIE.lock(%0, 3) {init = 0 : i32, sym_name = "of0_lock_3"}
// CHECK:     AIE.flow(%0, DMA : 0, %2, DMA : 0)
// CHECK:     %11 = AIE.buffer(%0) {sym_name = "of1_buff_0"} : memref<16xi32>
// CHECK:     %12 = AIE.buffer(%0) {sym_name = "of1_buff_1"} : memref<16xi32>
// CHECK:     %13 = AIE.lock(%0, 4) {init = 0 : i32, sym_name = "of1_lock_0"}
// CHECK:     %14 = AIE.lock(%0, 5) {init = 0 : i32, sym_name = "of1_lock_1"}
// CHECK:     %15 = AIE.buffer(%2) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
// CHECK:     %16 = AIE.buffer(%2) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
// CHECK:     %17 = AIE.lock(%2, 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// CHECK:     %18 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
// CHECK:     %19 = AIE.mem(%0) {
// CHECK:       %21 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%13, Acquire, 1)
// CHECK:       AIE.dmaBd(<%11 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%13, Release, 0)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%14, Acquire, 1)
// CHECK:       AIE.dmaBd(<%12 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%14, Release, 0)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %20 = AIE.mem(%2) {
// CHECK:       %21 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%17, Acquire, 0)
// CHECK:       AIE.dmaBd(<%15 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%17, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%18, Acquire, 0)
// CHECK:       AIE.dmaBd(<%16 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%18, Release, 1)
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
      AIE.objectFifo @of0 (%tile12, {%tile13}, 4 : i32) : !AIE.objectFifo<memref<16xi32>>

      // In the non-adjacent memory case, the number of elements depends on the max amount acquired by
      // the processes running on each core (here nothing is specified so it cannot be derived).
      AIE.objectFifo @of1 (%tile12, {%tile33}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>
   }
}
