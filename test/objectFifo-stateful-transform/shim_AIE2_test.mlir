//===- shim_AIE2_test.mlir --------------------------*- MLIR -*-===//
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

// CHECK: module @shim_AIE2 {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(2, 2)
// CHECK:     %1 = AIE.tile(2, 0)
// CHECK:     AIE.flow(%1, DMA : 0, %0, DMA : 0)
// CHECK:     %2 = AIE.lock(%1, 0) {init = 1 : i32, sym_name = "of_in_prod_lock"}
// CHECK:     %3 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "of_in_cons_lock"}
// CHECK:     %4 = AIE.buffer(%0) {sym_name = "of_in_cons_buff_0"} : memref<16xi32>
// CHECK:     %5 = AIE.buffer(%0) {sym_name = "of_in_cons_buff_1"} : memref<16xi32>
// CHECK:     %6 = AIE.lock(%0, 0) {init = 2 : i32, sym_name = "of_in_cons_prod_lock"}
// CHECK:     %7 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "of_in_cons_cons_lock"}
// CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
// CHECK:     %8 = AIE.buffer(%0) {sym_name = "of_out_buff_0"} : memref<16xi32>
// CHECK:     %9 = AIE.buffer(%0) {sym_name = "of_out_buff_1"} : memref<16xi32>
// CHECK:     %10 = AIE.lock(%0, 2) {init = 2 : i32, sym_name = "of_out_prod_lock"}
// CHECK:     %11 = AIE.lock(%0, 3) {init = 0 : i32, sym_name = "of_out_cons_lock"}
// CHECK:     %12 = AIE.lock(%1, 2) {init = 1 : i32, sym_name = "of_out_cons_prod_lock"}
// CHECK:     %13 = AIE.lock(%1, 3) {init = 0 : i32, sym_name = "of_out_cons_cons_lock"}
// CHECK:     %14 = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:     %15 = AIE.external_buffer {sym_name = "ext_buffer_out"} : memref<64xi32>
// CHECK:     AIE.shimDMAAllocation(@of_in, MM2S, 0, 2)
// CHECK:     %16 = AIE.shimDMA(%1) {
// CHECK:       %18 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       AIE.useLock(%3, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%14 : memref<64xi32>, 0, 64>, 0)
// CHECK:       AIE.useLock(%2, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       %19 = AIE.dmaStart(S2MM, 0, ^bb3, ^bb4)
// CHECK:     ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:       AIE.useLock(%12, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%15 : memref<64xi32>, 0, 64>, 0)
// CHECK:       AIE.useLock(%13, Release, 1)
// CHECK:       AIE.nextBd ^bb3
// CHECK:     ^bb4:  // pred: ^bb2
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     AIE.shimDMAAllocation(@of_out, S2MM, 0, 2)
// CHECK:     %17 = AIE.mem(%0) {
// CHECK:       %18 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%7, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%5 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%7, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb3:  // pred: ^bb0
// CHECK:       %19 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
// CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%8 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%10, Release, 1)
// CHECK:       AIE.nextBd ^bb5
// CHECK:     ^bb5:  // pred: ^bb4
// CHECK:       AIE.useLock(%11, AcquireGreaterEqual, 1)
// CHECK:       AIE.dmaBd(<%9 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%10, Release, 1)
// CHECK:       AIE.nextBd ^bb4
// CHECK:     ^bb6:  // pred: ^bb3
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @shim_AIE2 {
   AIE.device(xcve2302) {
      %tile22 = AIE.tile(2, 2)
      %tile20 = AIE.tile(2, 0)

      AIE.objectFifo @of_in (%tile20, {%tile22}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>
      AIE.objectFifo @of_out (%tile22, {%tile20}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>

      %ext_buffer_in  = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
      %ext_buffer_out  = AIE.external_buffer {sym_name = "ext_buffer_out"}: memref<64xi32>
      AIE.objectFifo.registerExternalBuffers(%tile20, %objFifo_in : !AIE.objectFifo<memref<16xi32>>, {%ext_buffer_in}) : (memref<64xi32>)
      AIE.objectFifo.registerExternalBuffers(%tile20, %objFifo_out : !AIE.objectFifo<memref<16xi32>>, {%ext_buffer_out}) : (memref<64xi32>)
   }
}
