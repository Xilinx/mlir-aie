//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Data Movement: AIE Core -> DMA -> AIE Core
// Pattern: Cyclostatic

// In this test, data is passed through L1 -> DMA -> L1 via an objectFifo, with 
// a cyclostatic pattern in the consumer.

// Producer pattern: {1}
// Consumer pattern: {1, 2, 1}

// The release and acquire always have the same number of elements in this test.

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

module @aie2_cyclostatic_dma {
    AIE.device(xcve2802) {

        %tile23 = AIE.tile(2, 3)  // producer tile
        %tile83 = AIE.tile(2, 8)  // consumer tile
        %buf83  = AIE.buffer(%tile83) {sym_name = "buf83"} : memref<17x4xi32>
        %lock83 = AIE.lock(%tile83, 0) { init = 0 : i32, sym_name = "lock83" }

        // ObjectFifo that can hold 4 memref<i32>s, populated by tile22 and
        // consumed by tile23
        %fifo = AIE.objectFifo.createObjectFifo(%tile23, {%tile83}, 4 : i32) {sym_name = "fifo"} : !AIE.objectFifo<memref<i32>>

        // Producer core
        %core23 = AIE.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i17 = arith.constant 17 : index

            %c11 = arith.constant 11 : i32
            %c22 = arith.constant 22 : i32
            %c33 = arith.constant 33 : i32
            %c44 = arith.constant 44 : i32

            scf.for %iter = %i0 to %i17 step %i1 {

                // Push 11
                %subview0 = AIE.objectFifo.acquire<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                memref.store %c11, %subview0_obj[] : memref<i32>
                AIE.objectFifo.release<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1)

                // Push 22 
                %subview1 = AIE.objectFifo.acquire<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview1_obj = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                memref.store %c22, %subview1_obj[] : memref<i32>
                AIE.objectFifo.release<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1)

                // Push 33 
                %subview2 = AIE.objectFifo.acquire<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview2_obj = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                memref.store %c33, %subview2_obj[] : memref<i32>
                AIE.objectFifo.release<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1)

                // Push 44 
                %subview3 = AIE.objectFifo.acquire<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview3_obj = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                memref.store %c44, %subview3_obj[] : memref<i32>
                AIE.objectFifo.release<Produce>(%fifo : !AIE.objectFifo<memref<i32>>, 1)

            }

            AIE.end
        }

        // Consumer core
        %core28 = AIE.core(%tile83) {
            // Consumer pattern: {1, 2, 1}
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index
            %i17 = arith.constant 17 : index
            %c2 = arith.constant 2 : i32
            %c3 = arith.constant 3 : i32

            scf.for %iter = %i0 to %i17 step %i1 {

                // Pop 1 object off queue; put it onto buf83 unaltered
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %v1 = memref.load %subview0_obj[] : memref<i32>
                memref.store %v1, %buf83[%iter, %i0] : memref<17x4xi32>
                AIE.objectFifo.release<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 1)

                // Pop 2 objects off queue; double them and put them into buf83
                %subview1 = AIE.objectFifo.acquire<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 2) : !AIE.objectFifoSubview<memref<i32>>
                %subview1_obj0 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %subview1_obj1 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %v2 = memref.load %subview1_obj0[] : memref<i32>
                %v3 = memref.load %subview1_obj1[] : memref<i32>
                %v4 = arith.muli %c2, %v2 : i32
                %v5 = arith.muli %c2, %v3 : i32
                memref.store %v4, %buf83[%iter, %i1] : memref<17x4xi32>
                memref.store %v5, %buf83[%iter, %i2] : memref<17x4xi32>
                AIE.objectFifo.release<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 2)

                // Pop 1 object off queue; triple it and put it into buf83
                %subview2 = AIE.objectFifo.acquire<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 1) : !AIE.objectFifoSubview<memref<i32>>
                %subview2_obj = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<i32>> -> memref<i32>
                %v6 = memref.load %subview2_obj[] : memref<i32>
                %v7 = arith.muli %c3, %v6 : i32
                memref.store %v7, %buf83[%iter, %i3] : memref<17x4xi32>
                AIE.objectFifo.release<Consume>(%fifo : !AIE.objectFifo<memref<i32>>, 1)

            }

            // Signal to host that we are done
            AIE.useLock(%lock83, "Release", 1)

            AIE.end
        }


    }
}
