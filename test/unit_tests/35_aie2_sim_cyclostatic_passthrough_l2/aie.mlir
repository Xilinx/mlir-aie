//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2013, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Pass through AIE -> Mem Tile (L2) -> AIE, with cyclostatic pattern.
// Producer pattern: {1}
// Consumer pattern: {2, 3, 3, 2}

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

// This test currently fails. The first 10 elements seem to be received
// correctly at the consumer, but after that things break down.

// Note that there is another bug if the objectFifoSize is increased to 16; In
// that case, the compilation succeeds, but the simulator reports:
// "At (1,1): Illegal BD id 24 used for channel number 0; BDs below 24 for even 
// channels."
// This seems like too many BDs are allocated -- something that could should be
// checked during compilation time.

// XFAIL: *

module @aie2_cyclostatic_passthrough_l2 {
    AIE.device(xcve2802) {

        %tile13 = AIE.tile(1, 3)  // producer tile
        %tile11 = AIE.tile(1, 1)  // mem tile
        %tile15 = AIE.tile(1, 5)  // consumer tile
        %buf13 = AIE.buffer(%tile13) {sym_name = "buf13"} : memref<1xi32> // iter_args workaround
        %buf15  = AIE.buffer(%tile15) {sym_name = "buf15"} : memref<4x11xi32>
        %lock15 = AIE.lock(%tile15, 0) { init = 0 : i32, sym_name = "lock15" }

        %fifo0 = AIE.objectFifo.createObjectFifo(%tile13, {%tile11}, 12 : i32) {sym_name = "fifo0"} : !AIE.objectFifo<memref<1xi32>>
        %fifo1 = AIE.objectFifo.createObjectFifo(%tile11, {%tile15}, 12 : i32) {sym_name = "fifo1"} : !AIE.objectFifo<memref<1xi32>>
        AIE.objectFifo.link({%fifo0}, {%fifo1}) : ({!AIE.objectFifo<memref<1xi32>>}, {!AIE.objectFifo<memref<1xi32>>})

        // Producer core
        // Writes iteration number onto stream
        %core13 = AIE.core(%tile13) {
            %i0   = arith.constant   0 : index
            %i1   = arith.constant   1 : index
            %i40  = arith.constant  40 : index
            %c1   = arith.constant   1 : i32
            %c0   = arith.constant   0 : i32

            memref.store %c0, %buf13[%i0] : memref<1xi32> // iter_args workaround

            scf.for %iter = %i0 to %i40 step %i1
            //    iter_args(%v = %c0) -> (i32)
            {
                %v = memref.load %buf13[%i0] : memref<1xi32> // iter_args workaround 
                %subview0 = AIE.objectFifo.acquire<Produce>(%fifo0 : !AIE.objectFifo<memref<1xi32>>, 1) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview0_obj = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                memref.store %v, %subview0_obj[%i0] : memref<1xi32>
                AIE.objectFifo.release<Produce>(%fifo0 : !AIE.objectFifo<memref<1xi32>>, 1)
                %v_next = arith.addi %c1, %v : i32
                // scf.yield %v_next : i32
                memref.store %v_next, %buf13[%i0] : memref<1xi32> // iter_args_workaround
            }

            AIE.end
        }

        // Consumer core
        %core15 = AIE.core(%tile15) {
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
            %c2  = arith.constant  2 : i32
            %c3  = arith.constant  3 : i32

            scf.for %iter = %i0 to %i4 step %i1 {
                
                // 2
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview0_obj0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview0_obj1 = AIE.objectFifo.subview.access %subview0[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v0_0 = memref.load %subview0_obj0[%i0] : memref<1xi32>
                %v0_1 = memref.load %subview0_obj1[%i0] : memref<1xi32>
                memref.store %v0_0, %buf15[%iter, %i0] : memref<4x11xi32>
                memref.store %v0_1, %buf15[%iter, %i1] : memref<4x11xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2)

                // 3
                %subview1 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview1_obj0 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview1_obj1 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview1_obj2 = AIE.objectFifo.subview.access %subview1[2] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v1_0 = memref.load %subview1_obj0[%i0] : memref<1xi32>
                %v1_1 = memref.load %subview1_obj1[%i0] : memref<1xi32>
                %v1_2 = memref.load %subview1_obj2[%i0] : memref<1xi32>
                memref.store %v1_0, %buf15[%iter, %i2] : memref<4x11xi32>
                memref.store %v1_1, %buf15[%iter, %i3] : memref<4x11xi32>
                memref.store %v1_2, %buf15[%iter, %i4] : memref<4x11xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3)

                // 3
                %subview2 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview2_obj0 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview2_obj1 = AIE.objectFifo.subview.access %subview2[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview2_obj2 = AIE.objectFifo.subview.access %subview2[2] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v2_0 = memref.load %subview2_obj0[%i0] : memref<1xi32>
                %v2_1 = memref.load %subview2_obj1[%i0] : memref<1xi32>
                %v2_2 = memref.load %subview2_obj2[%i0] : memref<1xi32>
                memref.store %v2_0, %buf15[%iter, %i5] : memref<4x11xi32>
                memref.store %v2_1, %buf15[%iter, %i6] : memref<4x11xi32>
                memref.store %v2_2, %buf15[%iter, %i7] : memref<4x11xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3)

                // 2
                %subview3 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview3_obj0 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview3_obj1 = AIE.objectFifo.subview.access %subview3[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v3_0 = memref.load %subview3_obj0[%i0] : memref<1xi32>
                %v3_1 = memref.load %subview3_obj1[%i0] : memref<1xi32>
                memref.store %v3_0, %buf15[%iter, %i8] : memref<4x11xi32>
                memref.store %v3_1, %buf15[%iter, %i9] : memref<4x11xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2)

            }

            // Signal to host that we are done
            AIE.useLock(%lock15, "Release", 1)

            AIE.end
        }

    }
}
