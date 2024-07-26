//===- AIE2_cyclostatic_dma.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2013, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Data Movement: AIE Core -> Core DMA -> Mem Tile -> Shim Tile DMA -> DDR
// Pattern: Static

// Producer AIE core sends data straight to host DDR via objectfifo.

// RUN: make -f %S/Makefile && %S/build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

// This test currently does not pass. The necessary 
//    mlir_aie_configure_shimdma_30(_xaie);
// function does not get generated in aie_inc.cpp,
// thus failing compilation of the host code.

// XFAIL: *

module @aie2_l1_ddr {
    aie.device(xcve2802) {

        %tile30 = aie.tile(3, 0)  // shim tile
        %tile31 = aie.tile(3, 1)  // mem tile
        %tile33 = aie.tile(3, 3)  // consumer tile
        %buf33  = aie.buffer(%tile33) {sym_name = "buf33"} : memref<i32>   // iter_args workaround
        %lock33 = aie.lock(%tile33, 0) { init = 0 : i32, sym_name = "lock33" }
        %extbuf0 = aie.external_buffer {sym_name = "extbuf0"} : memref<1xi32>
        %extbuf1 = aie.external_buffer {sym_name = "extbuf1"} : memref<1xi32>

        aie.objectfifo @fifo0 (%tile33, {%tile31}, 12 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo @fifo1 (%tile31, {%tile30}, 12 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo.link [@fifo0] -> [@fifo1] ()
        aie.objectfifo.register_external_buffers @fifo0 (%tile30, {%extbuf0, %extbuf1}) : (memref<1xi32>, memref<1xi32>)

        // Producer core
        %core33 = aie.core(%tile33) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i40 = arith.constant 40 : index
            %c0  = arith.constant  0 : i32
            %c1  = arith.constant  1 : i32

            memref.store %c0, %buf33[] : memref<i32>  // iter_args workaround

            scf.for %iter = %i0 to %i40 step %i1 
            // iter_args(%v = %c0) -> (i32)
            {
                %v = memref.load %buf33[] : memref<i32>  // iter_args workaround

                // Push value into objectfifo
                %subview0 = aie.objectfifo.acquire @fifo0 (Produce, 2) : !aie.objectfifosubview<memref<1xi32>>
                %subview0_obj0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
                memref.store %v, %subview0_obj0[%i0] : memref<1xi32>
                aie.objectfifo.release @fifo0 (Produce, 2)

                %v_next = arith.addi %v, %c1 : i32
                memref.store %v_next, %buf33[] : memref<i32>  // iter_args workaround
                // scf.yield %v_next : i32
            }

            // Signal to host that we are done
            aie.use_lock(%lock33, "Release", 1)

            aie.end
        }

    }
}
