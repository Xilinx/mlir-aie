//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2013, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Data Movement: DDR -> Shim Tile DMA -> Core DMA -> AIE Core
// Pattern: Cyclostatic

// Pass through host DDR (via shim tile) -> Mem Tile (L2) -> AIE, with 
// cyclostatic consumer pattern in AIE core.

// Consumer pattern: {2, 3, 3, 2}

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

module @aie2_cyclostatic_passthrough_ddr_mem_l1 {
    AIE.device(xcve2802) {

        %tile30 = AIE.tile(3, 0)  // shim tile
        %tile31 = AIE.tile(3, 1)  // mem tile
        %tile33 = AIE.tile(3, 3)  // consumer tile
        %buf33  = AIE.buffer(%tile33) {sym_name = "buf33"} : memref<4x10xi32>
        %lock33 = AIE.lock(%tile33, 0) { init = 0 : i32, sym_name = "lock33" }
        %extbuf0 = AIE.external_buffer {sym_name = "extbuf0"} : memref<1xi32>
        %extbuf1 = AIE.external_buffer {sym_name = "extbuf1"} : memref<1xi32>

        %fifo0 = AIE.objectFifo.createObjectFifo(%tile30, {%tile31}, 12 : i32) {sym_name = "fifo0"} : !AIE.objectFifo<memref<1xi32>>
        %fifo1 = AIE.objectFifo.createObjectFifo(%tile31, {%tile33}, 12 : i32) {sym_name = "fifo1"} : !AIE.objectFifo<memref<1xi32>>
        AIE.objectFifo.link({%fifo0}, {%fifo1}) : ({!AIE.objectFifo<memref<1xi32>>}, {!AIE.objectFifo<memref<1xi32>>})
        AIE.objectFifo.registerExternalBuffers(%tile30, %fifo0 : !AIE.objectFifo<memref<1xi32>>, {%extbuf0, %extbuf1}) : (memref<1xi32>, memref<1xi32>)

        // Consumer core
        %core33 = AIE.core(%tile33) {
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
                
                // consume 2
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview0_obj0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview0_obj1 = AIE.objectFifo.subview.access %subview0[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v0_0 = memref.load %subview0_obj0[%i0] : memref<1xi32>
                %v0_1 = memref.load %subview0_obj1[%i0] : memref<1xi32>
                memref.store %v0_0, %buf33[%iter, %i0] : memref<4x10xi32>
                memref.store %v0_1, %buf33[%iter, %i1] : memref<4x10xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2)

                // consume 3
                %subview1 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview1_obj0 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview1_obj1 = AIE.objectFifo.subview.access %subview1[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview1_obj2 = AIE.objectFifo.subview.access %subview1[2] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v1_0 = memref.load %subview1_obj0[%i0] : memref<1xi32>
                %v1_1 = memref.load %subview1_obj1[%i0] : memref<1xi32>
                %v1_2 = memref.load %subview1_obj2[%i0] : memref<1xi32>
                memref.store %v1_0, %buf33[%iter, %i2] : memref<4x10xi32>
                memref.store %v1_1, %buf33[%iter, %i3] : memref<4x10xi32>
                memref.store %v1_2, %buf33[%iter, %i4] : memref<4x10xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3)

                // consume 3
                %subview2 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview2_obj0 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview2_obj1 = AIE.objectFifo.subview.access %subview2[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview2_obj2 = AIE.objectFifo.subview.access %subview2[2] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v2_0 = memref.load %subview2_obj0[%i0] : memref<1xi32>
                %v2_1 = memref.load %subview2_obj1[%i0] : memref<1xi32>
                %v2_2 = memref.load %subview2_obj2[%i0] : memref<1xi32>
                memref.store %v2_0, %buf33[%iter, %i5] : memref<4x10xi32>
                memref.store %v2_1, %buf33[%iter, %i6] : memref<4x10xi32>
                memref.store %v2_2, %buf33[%iter, %i7] : memref<4x10xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 3)

                // consume 2
                %subview3 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview3_obj0 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %subview3_obj1 = AIE.objectFifo.subview.access %subview3[1] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v3_0 = memref.load %subview3_obj0[%i0] : memref<1xi32>
                %v3_1 = memref.load %subview3_obj1[%i0] : memref<1xi32>
                memref.store %v3_0, %buf33[%iter, %i8] : memref<4x10xi32>
                memref.store %v3_1, %buf33[%iter, %i9] : memref<4x10xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 2)

            }

            // Signal to host that we are done
            AIE.useLock(%lock33, "Release", 1)

            AIE.end
        }

    }
}
