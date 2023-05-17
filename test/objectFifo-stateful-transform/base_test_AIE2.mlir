//===- base_test_AIE2.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Xilinx Inc.
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @elementGenerationAIE2 {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(1, 2)
// CHECK:     %1 = AIE.tile(1, 3)
// CHECK:     %2 = AIE.tile(3, 3)
// CHECK:     %3 = AIE.buffer(%0) {sym_name = "of0_buff_0"} : memref<16xi32>
// CHECK:     %4 = AIE.buffer(%0) {sym_name = "of0_buff_1"} : memref<16xi32>
// CHECK:     %5 = AIE.buffer(%0) {sym_name = "of0_buff_2"} : memref<16xi32>
// CHECK:     %6 = AIE.buffer(%0) {sym_name = "of0_buff_3"} : memref<16xi32>
// CHECK:     %7 = AIE.lock(%0, 0) {init = 4 : i32, sym_name = "of0_prod_lock"}
// CHECK:     %8 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of0_cons_lock"}
// CHECK:     AIE.flow(%0, DMA : 0, %2, DMA : 0)
// CHECK:   }
// CHECK: }

module @elementGenerationAIE2 {
 AIE.device(xcve2302) {
    %tile12 = AIE.tile(1, 2)
    %tile13 = AIE.tile(1, 3)
    %tile33 = AIE.tile(3, 3)

    // In the shared memory case, the number of elements does not change.
    %objFifo0 = AIE.objectFifo.createObjectFifo(%tile12, {%tile13}, 4) : !AIE.objectFifo<memref<16xi32>>

    // In the non-adjacent memory case, the number of elements depends on the max amount acquired by
    // the processes running on each core (here nothing is specified so it cannot be derived).
    %objFifo1 = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>
 }
}
