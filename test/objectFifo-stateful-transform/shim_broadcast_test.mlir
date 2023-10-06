//===- shim_broadcast_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 3rd 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: module @shim_broadcast {
//CHECK:   AIE.device(xcve2302) {
//CHECK:     %0 = AIE.tile(2, 0)
//CHECK:     %1 = AIE.tile(2, 2)
//CHECK:     %2 = AIE.tile(2, 3)
//CHECK:     %3 = AIE.tile(3, 3)
//CHECK:     %4 = AIE.buffer(%1) {sym_name = "of_in_0_cons_buff_0"} : memref<16xi32>
//CHECK:     %5 = AIE.buffer(%1) {sym_name = "of_in_0_cons_buff_1"} : memref<16xi32>
//CHECK:     %6 = AIE.lock(%1, 0) {init = 2 : i32, sym_name = "of_in_0_cons_prod_lock"}
//CHECK:     %7 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "of_in_0_cons_cons_lock"}
//CHECK:     %8 = AIE.buffer(%2) {sym_name = "of_in_1_cons_buff_0"} : memref<16xi32>
//CHECK:     %9 = AIE.buffer(%2) {sym_name = "of_in_1_cons_buff_1"} : memref<16xi32>
//CHECK:     %10 = AIE.lock(%2, 0) {init = 2 : i32, sym_name = "of_in_1_cons_prod_lock"}
//CHECK:     %11 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "of_in_1_cons_cons_lock"}
//CHECK:     %12 = AIE.buffer(%3) {sym_name = "of_in_2_cons_buff_0"} : memref<16xi32>
//CHECK:     %13 = AIE.buffer(%3) {sym_name = "of_in_2_cons_buff_1"} : memref<16xi32>
//CHECK:     %14 = AIE.lock(%3, 0) {init = 2 : i32, sym_name = "of_in_2_cons_prod_lock"}
//CHECK:     %15 = AIE.lock(%3, 1) {init = 0 : i32, sym_name = "of_in_2_cons_cons_lock"}
//CHECK:     %16 = AIE.lock(%0, 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
//CHECK:     %17 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
//CHECK:     AIE.flow(%0, DMA : 0, %3, DMA : 0)
//CHECK:     AIE.flow(%0, DMA : 0, %2, DMA : 0)
//CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
//CHECK:     %18 = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
//CHECK:     AIE.shimDMAAllocation @of_in(MM2S, 0, 2)
//CHECK:     %19 = AIE.shimDMA(%0) {
//CHECK:       %23 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
//CHECK:       AIE.useLock(%17, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%18 : memref<64xi32>, 0, 64>, 0)
//CHECK:       AIE.useLock(%16, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb2:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %20 = AIE.mem(%1) {
//CHECK:       %23 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%5 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%7, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %21 = AIE.mem(%2) {
//CHECK:       %23 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%10, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%8 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%11, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%10, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%9 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%11, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %22 = AIE.mem(%3) {
//CHECK:       %23 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%12 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%15, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%14, AcquireGreaterEqual, 1)
//CHECK:       AIE.dmaBd(<%13 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%15, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }

module @shim_broadcast {
   AIE.device(xcve2302) {
      %tile20 = AIE.tile(2, 0)
      %tile22 = AIE.tile(2, 2)
      %tile23 = AIE.tile(2, 3)
      %tile33 = AIE.tile(3, 3)

      AIE.objectFifo @of_in (%tile20, {%tile22, %tile23, %tile33}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>

      %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      AIE.objectFifo.registerExternalBuffers @of_in (%tile20, {%ext_buffer_in}) : (memref<64xi32>)
   }
}
