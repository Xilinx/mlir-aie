//===- link_test_AIE1.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: May 9th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

//CHECK: module @link_AIE1 {
//CHECK:   AIE.device(xcvc1902) {
//CHECK:     memref.global "public" @of2_cons : memref<16xi32>
//CHECK:     memref.global "public" @of2 : memref<16xi32>
//CHECK:     memref.global "public" @of1_cons : memref<16xi32>
//CHECK:     memref.global "public" @of1 : memref<16xi32>
//CHECK:     %0 = AIE.tile(2, 0)
//CHECK:     %1 = AIE.tile(2, 2)
//CHECK:     %2 = AIE.tile(2, 4)
//CHECK:     %3 = AIE.buffer(%2) {sym_name = "of2_cons_buff_0"} : memref<16xi32>
//CHECK:     %4 = AIE.buffer(%2) {sym_name = "of2_cons_buff_1"} : memref<16xi32>
//CHECK:     %5 = AIE.lock(%2, 0) {init = 0 : i32, sym_name = "of2_cons_lock_0"}
//CHECK:     %6 = AIE.lock(%2, 1) {init = 0 : i32, sym_name = "of2_cons_lock_1"}
//CHECK:     %7 = AIE.buffer(%1) {sym_name = "of1_cons_buff_0"} : memref<16xi32>
//CHECK:     %8 = AIE.buffer(%1) {sym_name = "of1_cons_buff_1"} : memref<16xi32>
//CHECK:     %9 = AIE.lock(%1, 0) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
//CHECK:     %10 = AIE.lock(%1, 1) {init = 0 : i32, sym_name = "of1_cons_lock_1"}
//CHECK:     %11 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "of1_lock_0"}
//CHECK:     AIE.flow(%0, DMA : 0, %1, DMA : 0)
//CHECK:     AIE.flow(%1, DMA : 0, %2, DMA : 0)
//CHECK:     %12 = AIE.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32>
//CHECK:     AIE.shimDMAAllocation @of1(MM2S, 0, 2)
//CHECK:     %13 = AIE.shimDMA(%0) {
//CHECK:       %16 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
//CHECK:       AIE.useLock(%11, Acquire, 1)
//CHECK:       AIE.dmaBd(<%12 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%11, Release, 0)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb2:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %14 = AIE.mem(%1) {
//CHECK:       %16 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%9, Acquire, 0)
//CHECK:       AIE.dmaBd(<%7 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%9, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%10, Acquire, 0)
//CHECK:       AIE.dmaBd(<%8 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%10, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       %17 = AIE.dmaStart(MM2S, 0, ^bb4, ^bb6)
//CHECK:     ^bb4:  // 2 preds: ^bb3, ^bb5
//CHECK:       AIE.useLock(%9, Acquire, 1)
//CHECK:       AIE.dmaBd(<%7 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%9, Release, 0)
//CHECK:       AIE.nextBd ^bb5
//CHECK:     ^bb5:  // pred: ^bb4
//CHECK:       AIE.useLock(%10, Acquire, 1)
//CHECK:       AIE.dmaBd(<%8 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%10, Release, 0)
//CHECK:       AIE.nextBd ^bb4
//CHECK:     ^bb6:  // pred: ^bb3
//CHECK:       AIE.end
//CHECK:     }
//CHECK:     %15 = AIE.mem(%2) {
//CHECK:       %16 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb3)
//CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb2
//CHECK:       AIE.useLock(%5, Acquire, 0)
//CHECK:       AIE.dmaBd(<%3 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%5, Release, 1)
//CHECK:       AIE.nextBd ^bb2
//CHECK:     ^bb2:  // pred: ^bb1
//CHECK:       AIE.useLock(%6, Acquire, 0)
//CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
//CHECK:       AIE.useLock(%6, Release, 1)
//CHECK:       AIE.nextBd ^bb1
//CHECK:     ^bb3:  // pred: ^bb0
//CHECK:       AIE.end
//CHECK:     }
//CHECK:   }
//CHECK: }

module @link_AIE1 {
    AIE.device(xcvc1902) {
        %tile20 = AIE.tile(2, 0)
        %tile22 = AIE.tile(2, 2)
        %tile24 = AIE.tile(2, 4)
        
        AIE.objectFifo @of1 (%tile20, {%tile22}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>
        AIE.objectFifo @of2 (%tile22, {%tile24}, 2 : i32) : !AIE.objectFifo<memref<16xi32>>

        AIE.objectFifo.link [@of1] -> [@of2] ()

        %ext_buff_in = AIE.external_buffer {sym_name = "ext_buff_in"} : memref<16xi32> 
        AIE.objectFifo.registerExternalBuffers @of1 (%tile20, {%ext_buff_in}) : (memref<16xi32>)
    }
}
