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
// Pattern: Static

// Pass through host DDR (via shim tile) -> aie.

// RUN: make -f %S/Makefile && %S/build/aie.mlir.prj/aiesim.sh | FileCheck %s
// CHECK: AIE2 ISS
// CHECK: PASS!

module @aie2_cyclostatic_passthrough_ddr_l2 {
    aie.device(xcve2802) {

        %tile30 = aie.tile(3, 0)  // shim tile
        %tile31 = aie.tile(3, 1)  // mem tile
        %tile33 = aie.tile(3, 3)  // consumer tile
        %buf33  = aie.buffer(%tile33) {sym_name = "buf33"} : memref<40xi32>
        %lock33 = aie.lock(%tile33, 0) { init = 0 : i32, sym_name = "lock33" }
        %extbuf0 = aie.external_buffer {sym_name = "extbuf0"} : memref<1xi32>
        %extbuf1 = aie.external_buffer {sym_name = "extbuf1"} : memref<1xi32

        aie.objectfifo @fifo0 (%tile30, {%tile33}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
        aie.objectfifo.register_external_buffers @fifo0 (%tile30, {%extbuf0, %extbuf1}) : (memref<1xi32>, memref<1xi32>)

        // Consumer core
        %core33 = aie.core(%tile33) {
            %i0  = arith.constant  0 : index
            %i1  = arith.constant  1 : index
            %i40 = arith.constant 40 : index

            scf.for %iter = %i0 to %i40 step %i1 {
                
                %subview0 = aie.objectfifo.acquire @fifo0 (Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
                %subview0_obj0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
                %v0_0 = memref.load %subview0_obj0[%i0] : memref<1xi32>
                memref.store %v0_0, %buf33[%iter] : memref<40xi32>
                aie.objectfifo.release @fifo0 (Consume, 1)

            }

            // Signal to host that we are done
            aie.use_lock(%lock33, "Release", 1)

            aie.end
        }

    }
}
