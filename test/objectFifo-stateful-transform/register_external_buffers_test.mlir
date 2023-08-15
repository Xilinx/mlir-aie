//===- register_external_buffers_test.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: January 27th 2023
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @register_external_buffers {
// CHECK:   AIE.device(xcvc1902) {
// CHECK:     %0 = AIE.tile(7, 1)
// CHECK:     %1 = AIE.tile(7, 0)
// CHECK:     memref.global "public" @ext_of : memref<16xi32>
// CHECK:     AIE.flow(%1, DMA : 0, %0, DMA : 0)
// CHECK:     memref.global "public" @ext_of_cons : memref<16xi32>
// CHECK:     %2 = AIE.lock(%1, 0) {init = 0 : i32, sym_name = "ext_of_lock_0"}
// CHECK:     %3 = AIE.buffer(%0) {sym_name = "ext_of_cons_buff_0"} : memref<16xi32>
// CHECK:     %4 = AIE.buffer(%0) {sym_name = "ext_of_cons_buff_1"} : memref<16xi32>
// CHECK:     %5 = AIE.buffer(%0) {sym_name = "ext_of_cons_buff_2"} : memref<16xi32>
// CHECK:     %6 = AIE.lock(%0, 0) {init = 0 : i32, sym_name = "ext_of_cons_lock_0"}
// CHECK:     %7 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "ext_of_cons_lock_1"}
// CHECK:     %8 = AIE.lock(%0, 2) {init = 0 : i32, sym_name = "ext_of_cons_lock_2"}
// CHECK:     %9 = AIE.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
// CHECK:     AIE.shimDMAAllocation(@ext_of, MM2S, 0, 7)
// CHECK:     func.func @some_work(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
// CHECK:       return
// CHECK:     }
// CHECK:     %10 = AIE.core(%0) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c12 = arith.constant 12 : index
// CHECK:       AIE.useLock(%6, Acquire, 1)
// CHECK:       AIE.useLock(%7, Acquire, 1)
// CHECK:       func.call @some_work(%3, %4) : (memref<16xi32>, memref<16xi32>) -> ()
// CHECK:       AIE.useLock(%6, Release, 0)
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %11 = AIE.shimDMA(%1) {
// CHECK:       %13 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:       AIE.useLock(%2, Acquire, 1)
// CHECK:       AIE.dmaBd(<%9 : memref<64xi32>, 0, 64>, 0)
// CHECK:       AIE.useLock(%2, Release, 0)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb2:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %12 = AIE.mem(%0) {
// CHECK:       %13 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb4)
// CHECK:     ^bb1:  // 2 preds: ^bb0, ^bb3
// CHECK:       AIE.useLock(%6, Acquire, 0)
// CHECK:       AIE.dmaBd(<%3 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%6, Release, 1)
// CHECK:       AIE.nextBd ^bb2
// CHECK:     ^bb2:  // pred: ^bb1
// CHECK:       AIE.useLock(%7, Acquire, 0)
// CHECK:       AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%7, Release, 1)
// CHECK:       AIE.nextBd ^bb3
// CHECK:     ^bb3:  // pred: ^bb2
// CHECK:       AIE.useLock(%8, Acquire, 0)
// CHECK:       AIE.dmaBd(<%5 : memref<16xi32>, 0, 16>, 0)
// CHECK:       AIE.useLock(%8, Release, 1)
// CHECK:       AIE.nextBd ^bb1
// CHECK:     ^bb4:  // pred: ^bb0
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @register_external_buffers {
 AIE.device(xcvc1902) {
    %tile71 = AIE.tile(7, 1)
    %tile70 = AIE.tile(7, 0)

    AIE.objectFifo @ext_of (%tile70, {%tile71}, 3 : i32) : !AIE.objectFifo<memref<16xi32>>

    %ext_buffer_in = AIE.external_buffer {sym_name = "ext_buffer_in"}: memref<64xi32>
    AIE.objectFifo.registerExternalBuffers @ext_of (%tile70, {%ext_buffer_in}) : (memref<64xi32>)

    func.func @some_work(%a : memref<16xi32>, %b : memref<16xi32>) -> () {
        return
    }

    %core71 = AIE.core(%tile71) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        %subview = AIE.objectFifo.acquire @ext_of (Consume, 2) : !AIE.objectFifoSubview<memref<16xi32>>
        %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        %elem1 = AIE.objectFifo.subview.access %subview[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
        func.call @some_work(%elem0, %elem1) : (memref<16xi32>, memref<16xi32>) -> ()
        AIE.objectFifo.release @ext_of (Consume, 1)
        
        AIE.end
    }
 }
}
