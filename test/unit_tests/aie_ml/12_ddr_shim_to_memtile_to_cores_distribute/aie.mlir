//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2013, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Data Movement: DDR -> Shim Tile DMA -> Mem Tile ----> Core DMA -> AIE Core
//                                                  \--> Core DMA -> AIE Core
//                                                   \-> Core DMA -> AIE Core 
// (Distribute)
// Pattern: Static

// The host in this core writes the sequence 0, 1000, 2000, 1, 1001, 2001,
// 2, 1002, 2002, ... (i.e. buf[i] = i%3*1000 + i) into an external buffer.
// The external buffer is routed via stream into a mem tile. From there, it is
// distributed round-robin to each of three cores. With everything behaving
// correctly, core 1 receives values < 1000, core 2 receives values < 2000, and
// core 3 receives 2000 <= values < 3000. 
// The consumer cores simply write the values they receive into a buffer in
// their L1 memory. The host checks that this memory contains the right values
// upon completion of the cores.

// RUN: make && ./build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

// This test currently fails during the latter iterations of the first consumer
// core. When the iteration count in the cores is < 33, the test passes. With
// an iteration count of 33, at the 32nd iteration, core 1 should receive value
// 0031; however, it appears to read value 0033. The other two cores correctly
// receive 1031 and 2031, respectively. With iterations > 33, the test hangs;
// the host times out while waiting to acquire the lock for core 1 to be
// finished (lock33).

// XFAIL: *

module @aie2_cyclostatic_passthrough_ddr_l2 {
    AIE.device(xcve2802) {

        %tile30 = AIE.tile(3, 0)  // shim tile
        %tile31 = AIE.tile(3, 1)  // mem tile
        %tile33 = AIE.tile(3, 3)  // consumer tile 1
        %tile34 = AIE.tile(3, 4)  // consumer tile 2
        %tile35 = AIE.tile(3, 5)  // consumer tile 3
        %buf33  = AIE.buffer(%tile33) {sym_name = "buf33"} : memref<40xi32>
        %buf34  = AIE.buffer(%tile34) {sym_name = "buf34"} : memref<40xi32>
        %buf35  = AIE.buffer(%tile35) {sym_name = "buf35"} : memref<40xi32>
        %lock33 = AIE.lock(%tile33, 0) { init = 0 : i32, sym_name = "lock33" }
        %lock34 = AIE.lock(%tile34, 0) { init = 0 : i32, sym_name = "lock34" }
        %lock35 = AIE.lock(%tile35, 0) { init = 0 : i32, sym_name = "lock35" }
        %extbuf0 = AIE.external_buffer {sym_name = "extbuf0"} : memref<3xi32>
        %extbuf1 = AIE.external_buffer {sym_name = "extbuf1"} : memref<3xi32>

        %fifo0 = AIE.objectFifo.createObjectFifo(%tile30, {%tile31}, 2 : i32) {sym_name = "fifo0"} : !AIE.objectFifo<memref<3xi32>>
        AIE.objectFifo.registerExternalBuffers(%tile30, %fifo0 : !AIE.objectFifo<memref<3xi32>>, {%extbuf0, %extbuf1}) : (memref<3xi32>, memref<3xi32>)
        %fifo1 = AIE.objectFifo.createObjectFifo(%tile31, {%tile33}, 2 : i32) {sym_name = "fifo1"} : !AIE.objectFifo<memref<1xi32>>
        %fifo2 = AIE.objectFifo.createObjectFifo(%tile31, {%tile34}, 2 : i32) {sym_name = "fifo2"} : !AIE.objectFifo<memref<1xi32>>
        %fifo3 = AIE.objectFifo.createObjectFifo(%tile31, {%tile35}, 2 : i32) {sym_name = "fifo3"} : !AIE.objectFifo<memref<1xi32>>
        AIE.objectFifo.link({%fifo0}, {%fifo1, %fifo2, %fifo3}) : ({!AIE.objectFifo<memref<3xi32>>}, {!AIE.objectFifo<memref<1xi32>>, !AIE.objectFifo<memref<1xi32>>, !AIE.objectFifo<memref<1xi32>>})


        // Consumer core 1
        %core33 = AIE.core(%tile33) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i40 = arith.constant 33 : index

            scf.for %iter = %i0 to %i40 step %i1 {
                
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 1) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview0_obj0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v0_0 = memref.load %subview0_obj0[%i0] : memref<1xi32>
                memref.store %v0_0, %buf33[%iter] : memref<40xi32>
                AIE.objectFifo.release<Consume>(%fifo1 : !AIE.objectFifo<memref<1xi32>>, 1)

            }

            // Signal to host that we are done
            aie.use_lock(%lock33, "Release", 1)

            AIE.end
        }

        // Consumer core 2
        %core34 = AIE.core(%tile34) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i40 = arith.constant 33 : index

            scf.for %iter = %i0 to %i40 step %i1 {
                
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo2 : !AIE.objectFifo<memref<1xi32>>, 1) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview0_obj0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v0_0 = memref.load %subview0_obj0[%i0] : memref<1xi32>
                memref.store %v0_0, %buf34[%iter] : memref<40xi32>
                AIE.objectFifo.release<Consume>(%fifo2 : !AIE.objectFifo<memref<1xi32>>, 1)

            }

            // Signal to host that we are done
            aie.use_lock(%lock34, "Release", 1)

            AIE.end
        }

        // Consumer core 3
        %core35 = AIE.core(%tile35) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i40 = arith.constant 33 : index

            scf.for %iter = %i0 to %i40 step %i1 {
                
                %subview0 = AIE.objectFifo.acquire<Consume>(%fifo3 : !AIE.objectFifo<memref<1xi32>>, 1) : !AIE.objectFifoSubview<memref<1xi32>>
                %subview0_obj0 = AIE.objectFifo.subview.access %subview0[0] : !AIE.objectFifoSubview<memref<1xi32>> -> memref<1xi32>
                %v0_0 = memref.load %subview0_obj0[%i0] : memref<1xi32>
                memref.store %v0_0, %buf35[%iter] : memref<40xi32>
                AIE.objectFifo.release<Consume>(%fifo3 : !AIE.objectFifo<memref<1xi32>>, 1)

            }

            // Signal to host that we are done
            aie.use_lock(%lock35, "Release", 1)

            AIE.end
        }
    }
}