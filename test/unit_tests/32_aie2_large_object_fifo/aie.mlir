//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This tests an objectFifo with a large number of objects (objects themselves
// are small).

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

// Currently, this test fails with a too large iteration number in core 28.
// I believe this has to do with the object fifo loop unrolling.
// XFAIL: *

module @aie2_cyclostatic_dma {
    AIE.device(xcve2802) {

        %tile23 = AIE.tile(2, 3)  // producer tile
        %tile28 = AIE.tile(2, 8)  // consumer tile
        %buf28  = AIE.buffer(%tile28) {sym_name = "buf28"} : memref<16x10xi32>
        %lock28 = AIE.lock(%tile28, 0) { init = 0 : i32, sym_name = "lock28" }

        %fifo = AIE.objectFifo.createObjectFifo(%tile23, {%tile28}, 20 : i32) {sym_name = "fifo"} : !AIE.objectFifo<memref<i32>>

        // Producer core
        %core23 = AIE.core(%tile23) {
            %i0   = arith.constant   0 : index
            %i1   = arith.constant   1 : index
            %i160 = arith.constant 160 : index
            %c7   = arith.constant   7 : i32

            scf.for %iter = %i0 to %i160 step %i1
            {
                %subview0 = AIE.objectFifo.acquire<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                memref.store %c7, %subview0_obj[] : memref<i32>
                AIE.objectFifo.release<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1)
            }

            AIE.end
        }

        // Consumer core
        %core28 = AIE.core(%tile28) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i2  = arith.constant  2 : index
            %i3  = arith.constant  3 : index
            %i4  = arith.constant  4 : index
            %i5  = arith.constant  5 : index
            %i6  = arith.constant  6 : index
            %i7  = arith.constant  7 : index
            %i8  = arith.constant  8 : index
            %i9  = arith.constant  9 : index
            %i16 = arith.constant  8 : index
            %c2  = arith.constant  2 : i32
            %c3  = arith.constant  3 : i32

            scf.for %iter = %i0 to %i16 step %i1 {
                
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 10) : !AIE.objectFifoSubview<memref<i32>>
                %subview0_obj0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj1 = AIE.objectFifo.subview.access %subview0[1] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj2 = AIE.objectFifo.subview.access %subview0[2] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj3 = AIE.objectFifo.subview.access %subview0[3] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj4 = AIE.objectFifo.subview.access %subview0[4] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj5 = AIE.objectFifo.subview.access %subview0[5] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj6 = AIE.objectFifo.subview.access %subview0[6] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj7 = AIE.objectFifo.subview.access %subview0[7] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj8 = AIE.objectFifo.subview.access %subview0[8] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview0_obj9 = AIE.objectFifo.subview.access %subview0[9] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %v0_0 = memref.load %subview0_obj0[] : memref<i32>
                %v0_1 = memref.load %subview0_obj1[] : memref<i32>
                %v0_2 = memref.load %subview0_obj2[] : memref<i32>
                %v0_3 = memref.load %subview0_obj3[] : memref<i32>
                %v0_4 = memref.load %subview0_obj4[] : memref<i32>
                %v0_5 = memref.load %subview0_obj5[] : memref<i32>
                %v0_6 = memref.load %subview0_obj6[] : memref<i32>
                %v0_7 = memref.load %subview0_obj7[] : memref<i32>
                %v0_8 = memref.load %subview0_obj8[] : memref<i32>
                %v0_9 = memref.load %subview0_obj9[] : memref<i32>
                memref.store %v0_0, %buf28[%iter, %i0] : memref<16x10xi32>
                memref.store %v0_1, %buf28[%iter, %i1] : memref<16x10xi32>
                memref.store %v0_2, %buf28[%iter, %i2] : memref<16x10xi32>
                memref.store %v0_3, %buf28[%iter, %i3] : memref<16x10xi32>
                memref.store %v0_4, %buf28[%iter, %i4] : memref<16x10xi32>
                memref.store %v0_5, %buf28[%iter, %i5] : memref<16x10xi32>
                memref.store %v0_6, %buf28[%iter, %i6] : memref<16x10xi32>
                memref.store %v0_7, %buf28[%iter, %i7] : memref<16x10xi32>
                memref.store %v0_8, %buf28[%iter, %i8] : memref<16x10xi32>
                memref.store %v0_9, %buf28[%iter, %i9] : memref<16x10xi32>
                AIE.objectFifo.release<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 10)

            }

            AIE.useLock(%lock28, "Release", 1)

            AIE.end
        }

    }
}
