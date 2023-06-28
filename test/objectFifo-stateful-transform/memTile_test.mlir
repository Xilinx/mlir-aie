//===- memTile_test.mlir --------------------------*- MLIR -*-===//
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

// CHECK: module @memTile {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(2, 1)
// CHECK:     %1 = AIE.tile(2, 2)
// CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
// CHECK:     %2 = AIE.buffer(%0) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:     %3 = AIE.buffer(%0) {sym_name = "of_buff_1"} : memref<16xi32>
// CHECK:     %4 = AIE.lock(%0, 0) {init = 2 : i32, sym_name = "of_prod_lock"}
// CHECK:     %5 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of_cons_lock"}
// CHECK:     %6 = AIE.buffer(%1) {sym_name = "of_cons_buff_0"} : memref<16xi32>
// CHECK:     %7 = AIE.buffer(%1) {sym_name = "of_cons_buff_1"} : memref<16xi32>
// CHECK:     %8 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "of_cons_prod_lock"}
// CHECK:     %9 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "of_cons_cons_lock"}
// CHECK:     %10 = AIE.memTileDMA(%0) {
// CHECK:       %12 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%5, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%2 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%4, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%5, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%3 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%4, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %11 = AIE.mem(%1) {
// CHECK:       %12 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%8, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%6 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%8, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%7 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @memTile {
 AIE.device(xcve2302) {
    %tile11 = AIE.tile(2, 1)
    %tile12 = AIE.tile(2, 2)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile11, {%tile12}, 2 : i32) {sym_name = "of"} : !AIE.objectFifo<memref<16xi32>>
 }
}
