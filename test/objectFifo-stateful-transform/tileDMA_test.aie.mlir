//===- tileDMA_test.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Date: September 22nd 2022
// 
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @tileDMA_channels {
// CHECK:   %0 = AIE.tile(1, 2)
// CHECK:   %1 = AIE.tile(3, 3)
// CHECK:   %2 = AIE.buffer(%0) : memref<16xi32>
// CHECK:   %3 = AIE.lock(%0, 0)
// CHECK:   %4 = AIE.buffer(%0) : memref<16xi32>
// CHECK:   %5 = AIE.lock(%0, 1)
// CHECK:   %6 = AIE.buffer(%0) : memref<16xi32>
// CHECK:   %7 = AIE.lock(%0, 2)
// CHECK:   AIE.flow(%0, DMA : 1, %1, DMA : 0)
// CHECK:   %8 = AIE.buffer(%0) {sym_name = "of_0_buff_0"} : memref<16xi32>
// CHECK:   %9 = AIE.lock(%0, 3) {sym_name = "of_0_lock_0"}
// CHECK:   %10 = AIE.buffer(%0) {sym_name = "of_0_buff_1"} : memref<16xi32>
// CHECK:   %11 = AIE.lock(%0, 4) {sym_name = "of_0_lock_1"}
// CHECK:   func.func @some_work(%arg0: memref<16xi32>) {
// CHECK:     return
// CHECK:   }
// CHECK:   %12 = AIE.core(%0) {
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c12 = arith.constant 12 : index
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     scf.for %arg0 = %c0 to %c12 step %c2 {
// CHECK:       AIE.useLock(%9, Acquire, 0)
// CHECK:       func.call @some_work(%8) : (memref<16xi32>) -> ()
// CHECK:       AIE.useLock(%9, Release, 1)
// CHECK:       AIE.useLock(%11, Acquire, 0)
// CHECK:       func.call @some_work(%10) : (memref<16xi32>) -> ()
// CHECK:       AIE.useLock(%11, Release, 1)
// CHECK:     }
// CHECK:     AIE.end
// CHECK:   }
// CHECK:   %13 = AIE.mem(%0) {
// CHECK:     %14 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
// CHECK:   ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:     AIE.useLock(%3, Acquire, 1)
// CHECK:     AIE.dmaBd(<%2 : memref<16xi32>, 0, 16>, 0)
// CHECK:     AIE.useLock(%3, Release, 0)
// CHECK:     AIE.nextBd ^bb2
// CHECK:   ^bb2:  // pred: ^bb1
// CHECK:     AIE.useLock(%5, Acquire, 1)
// CHECK:     AIE.dmaBd(<%4 : memref<16xi32>, 0, 16>, 0)
// CHECK:     AIE.useLock(%5, Release, 0)
// CHECK:     AIE.nextBd ^bb1
// CHECK:   ^bb3:  // pred: ^bb0
// CHECK:     %15 = AIE.dmaStart(S2MM, 0, ^bb4, ^bb5)
// CHECK:   ^bb4:  // 2 preds: ^bb3, ^bb4
// CHECK:     AIE.useLock(%7, Acquire, 0)
// CHECK:     AIE.dmaBd(<%6 : memref<16xi32>, 0, 16>, 0)
// CHECK:     AIE.useLock(%7, Release, 1)
// CHECK:     AIE.nextBd ^bb4
// CHECK:   ^bb5:  // pred: ^bb3
// CHECK:     %16 = AIE.dmaStart(MM2S, 1, ^bb6, ^bb8)
// CHECK:   ^bb6:  // 2 preds: ^bb5, ^bb7
// CHECK:     AIE.useLock(%9, Acquire, 1)
// CHECK:     AIE.dmaBd(<%8 : memref<16xi32>, 0, 16>, 0)
// CHECK:     AIE.useLock(%9, Release, 0)
// CHECK:     AIE.nextBd ^bb7
// CHECK:   ^bb7:  // pred: ^bb6
// CHECK:     AIE.useLock(%11, Acquire, 1)
// CHECK:     AIE.dmaBd(<%10 : memref<16xi32>, 0, 16>, 0)
// CHECK:     AIE.useLock(%11, Release, 0)
// CHECK:     AIE.nextBd ^bb6
// CHECK:   ^bb8:  // pred: ^bb5
// CHECK:     AIE.end
// CHECK:   }
// CHECK: }

module @tileDMA_channels {
 AIE.device(xcvc1902) {
    %tile12 = AIE.tile(1, 2)
    %tile33 = AIE.tile(3, 3)

    %buff0 = AIE.buffer(%tile12) : memref<16xi32>
    %lock0 = AIE.lock(%tile12, 0)
    %buff1 = AIE.buffer(%tile12) : memref<16xi32>
    %lock1 = AIE.lock(%tile12, 1)
    %buff2 = AIE.buffer(%tile12) : memref<16xi32>
    %lock2 = AIE.lock(%tile12, 2)

    %objFifo = AIE.objectFifo.createObjectFifo(%tile12, {%tile33}, 2) : !AIE.objectFifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
        return
    }

    %core12 = AIE.core(%tile12) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %height = arith.constant 12 : index

        scf.for %indexInHeight = %c0 to %height step %c1 {
            %subview = AIE.objectFifo.acquire<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %elem0 = AIE.objectFifo.subview.access %subview[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            func.call @some_work(%elem0) : (memref<16xi32>) -> ()
            AIE.objectFifo.release<Produce>(%objFifo : !AIE.objectFifo<memref<16xi32>>, 1)
        }
        
        AIE.end
    }

    %mem12 = AIE.mem(%tile12) {
        %dma1 = AIE.dmaStart(MM2S, 0, ^bb1, ^bb3)
    ^bb1:
        AIE.useLock(%lock0, Acquire, 1)
        AIE.dmaBd(<%buff0 : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock0, Release, 0)
        AIE.nextBd ^bb2
    ^bb2:
        AIE.useLock(%lock1, Acquire, 1)
        AIE.dmaBd(<%buff1 : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock1, Release, 0)
        AIE.nextBd ^bb1
    ^bb3:
        %dma2 = AIE.dmaStart(S2MM, 0, ^bb4, ^bb5)
    ^bb4:
        AIE.useLock(%lock2, Acquire, 0)
        AIE.dmaBd(<%buff2 : memref<16xi32>, 0, 16>, 0)
        AIE.useLock(%lock2, Release, 1)
        AIE.nextBd ^bb4
    ^bb5:
        AIE.end
    }
 }
}
