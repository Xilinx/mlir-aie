//===- cyclostatic_AIE2.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
// Date: July 10th 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: module @cyclostatic {
// CHECK:   AIE.device(xcve2302) {
// CHECK:     %0 = AIE.tile(1, 2)
// CHECK:     %1 = AIE.tile(2, 2)
// CHECK:     %2 = AIE.buffer(%0) {sym_name = "fifo0_buff_0"} : memref<16xi32>
// CHECK:     %3 = AIE.buffer(%0) {sym_name = "fifo0_buff_1"} : memref<16xi32>
// CHECK:     %4 = AIE.buffer(%0) {sym_name = "fifo0_buff_2"} : memref<16xi32>
// CHECK:     %5 = AIE.buffer(%0) {sym_name = "fifo0_buff_3"} : memref<16xi32>
// CHECK:     %6 = AIE.lock(%0, 0) {init = 4 : i32, sym_name = "fifo0_prod_lock"}
// CHECK:     %7 = AIE.lock(%0, 1) {init = 0 : i32, sym_name = "fifo0_cons_lock"}
// CHECK:     %8 = AIE.core(%0) {
// CHECK:       %c11_i32 = arith.constant 11 : i32
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c9 = arith.constant 9 : index
// CHECK:       %c8 = arith.constant 8 : index
// CHECK:       %c4 = arith.constant 4 : index
// CHECK:       scf.for %arg0 = %c0 to %c8 step %c4 {
// CHECK:         AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:         memref.store %c11_i32, %2[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%7, Release, 1)
// CHECK:         AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:         memref.store %c11_i32, %3[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%7, Release, 1)
// CHECK:         AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:         memref.store %c11_i32, %4[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%7, Release, 1)
// CHECK:         AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:         memref.store %c11_i32, %5[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%7, Release, 1)
// CHECK:       }
// CHECK:       AIE.useLock(%6, AcquireGreaterEqual, 1)
// CHECK:       memref.store %c11_i32, %2[%c0] : memref<16xi32>
// CHECK:       AIE.useLock(%7, Release, 1)
// CHECK:       AIE.end
// CHECK:     }
// CHECK:     %9 = AIE.core(%1) {
// CHECK:       %c0 = arith.constant 0 : index
// CHECK:       %c1 = arith.constant 1 : index
// CHECK:       %c9 = arith.constant 9 : index
// CHECK:       AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:       %10 = memref.load %2[%c0] : memref<16xi32>
// CHECK:       AIE.useLock(%6, Release, 1)
// CHECK:       %c8 = arith.constant 8 : index
// CHECK:       %c4 = arith.constant 4 : index
// CHECK:       scf.for %arg0 = %c0 to %c8 step %c4 {
// CHECK:         AIE.useLock(%7, AcquireGreaterEqual, 3)
// CHECK:         %16 = memref.load %3[%c0] : memref<16xi32>
// CHECK:         %17 = memref.load %4[%c0] : memref<16xi32>
// CHECK:         %18 = memref.load %5[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%6, Release, 1)
// CHECK:         AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:         %19 = memref.load %4[%c0] : memref<16xi32>
// CHECK:         %20 = memref.load %5[%c0] : memref<16xi32>
// CHECK:         %21 = memref.load %2[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%6, Release, 1)
// CHECK:         AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:         %22 = memref.load %5[%c0] : memref<16xi32>
// CHECK:         %23 = memref.load %2[%c0] : memref<16xi32>
// CHECK:         %24 = memref.load %3[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%6, Release, 1)
// CHECK:         AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:         %25 = memref.load %2[%c0] : memref<16xi32>
// CHECK:         %26 = memref.load %3[%c0] : memref<16xi32>
// CHECK:         %27 = memref.load %4[%c0] : memref<16xi32>
// CHECK:         AIE.useLock(%6, Release, 1)
// CHECK:       }
// CHECK:       AIE.useLock(%7, AcquireGreaterEqual, 1)
// CHECK:       %11 = memref.load %3[%c0] : memref<16xi32>
// CHECK:       %12 = memref.load %4[%c0] : memref<16xi32>
// CHECK:       %13 = memref.load %5[%c0] : memref<16xi32>
// CHECK:       AIE.useLock(%6, Release, 1)
// CHECK:       %14 = memref.load %4[%c0] : memref<16xi32>
// CHECK:       %15 = memref.load %5[%c0] : memref<16xi32>
// CHECK:       AIE.useLock(%6, Release, 2)
// CHECK:       AIE.end
// CHECK:     }
// CHECK:   }
// CHECK: }

module @cyclostatic {
    AIE.device(xcve2302) {
        %tile12 = AIE.tile(1, 2)
        %tile23 = AIE.tile(2, 2)

        %fifo0 = AIE.objectFifo.createObjectFifo(%tile12, {%tile23}, 4 : i32) {sym_name = "fifo0"} : !AIE.objectFifo<memref<16xi32>>

        %core12 = AIE.core(%tile12) {
            %v11 = arith.constant 11 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1 = AIE.objectFifo.acquire<Produce>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
                %subview1_obj = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                memref.store %v11, %subview1_obj[%c0] : memref<16xi32>
                AIE.objectFifo.release<Produce>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 1)
            }

            AIE.end
        }

        %core23 = AIE.core(%tile23) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            %subview0 = AIE.objectFifo.acquire<Consume>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 1) : !AIE.objectFifoSubview<memref<16xi32>>
            %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %v0 = memref.load %subview0_obj[%c0] : memref<16xi32>
            AIE.objectFifo.release<Consume>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 1)

            scf.for %indexInHeight = %c0 to %c9 step %c1 {
                %subview1 = AIE.objectFifo.acquire<Consume>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 3) : !AIE.objectFifoSubview<memref<16xi32>>
                %subview1_obj = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                %subview1_obj1 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                %subview1_obj2 = AIE.objectFifo.subview.access %subview1[2] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
                %v1 = memref.load %subview1_obj[%c0] : memref<16xi32>
                %v2 = memref.load %subview1_obj1[%c0] : memref<16xi32>
                %v3 = memref.load %subview1_obj2[%c0] : memref<16xi32>
                AIE.objectFifo.release<Consume>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 1)
            }

            %subview2 = AIE.objectFifo.acquire<Consume>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 2) : !AIE.objectFifoSubview<memref<16xi32>>
            %subview2_obj = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %subview2_obj1 = AIE.objectFifo.subview.access %subview2[1] : !AIE.objectFifoSubview<memref<16xi32>> -> memref<16xi32>
            %v4 = memref.load %subview2_obj[%c0] : memref<16xi32>
            %v5 = memref.load %subview2_obj1[%c0] : memref<16xi32>
            AIE.objectFifo.release<Consume>(%fifo0 : !AIE.objectFifo<memref<16xi32>>, 2)

            AIE.end
        }
    }
}
