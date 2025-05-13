//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// aiecc.py -j4 -%VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o tutorial-5.exe

// REQUIRES: valid_xchess_license
// RUN: make -f %S/Makefile
// RUN: %run_on_board ./tutorial-5.exe
// RUN: make -f %S/Makefile clean

// Declare this MLIR module. A wrapper that can contain all
// AIE tiles, buffers, and data movement
module @tutorial_5 {
    aie.device(xcvc1902) {
        // 1 tile in row 4 (col 3)
        // even rows have local memory to its left
        %tile34 = aie.tile(3, 4)

        // 1 tile in row 0 (col 7)
        // col 7, row 0 has access to a shim_dma
        %tile70 = aie.tile(7, 0)

        // Declare external buffers, which represent pointers to external memory locations.
        %ext_buf70_in  = aie.external_buffer {sym_name = "ddr_test_buffer_in"}: memref<256xi32>
        %ext_buf70_out = aie.external_buffer {sym_name = "ddr_test_buffer_out"}: memref<256xi32>

        // Declare an object FIFO between the producer shim tile (7,0) and consumer tile (3,4).
        // The size of the object FIFO, i.e. its number of elements, is 1.
        // Objects, i.e. allocated memory elements, have type memref<256xi32>.
        // These tiles do not share memory between them.
        aie.objectfifo @of_in (%tile70, {%tile34}, 1 : i32) : !aie.objectfifo<memref<256xi32>>

        // Declare an object FIFO between the producer tile (3,4) and consumer shim tile (7,0).
        aie.objectfifo @of_out (%tile34, {%tile70}, 1 : i32) : !aie.objectfifo<memref<256xi32>>

        // Register the external memory pointers to the object FIFOs.
        aie.objectfifo.register_external_buffers @of_in (%tile70, {%ext_buf70_in}) : (memref<256xi32>)
        aie.objectfifo.register_external_buffers @of_out (%tile70, {%ext_buf70_out}) : (memref<256xi32>)

        // Define core algorithm for tile(3,4) which reads value set by tile(1,4)
        // buf[5] = buf[3] + 100
        %core34 = aie.core(%tile34) {
            // Acquire a subview with one object from each object FIFO.
            // This is equivalent to acquiring an AIE lock before accessing an AIE buffer.
            // This core acquires objects both as a Consumer of one object FIFO and as a Producer of another:
            // this impacts the acquire values of the locks that are generated through the object FIFO lowering
            %inputSubview = aie.objectfifo.acquire @of_in (Consume, 1) : !aie.objectfifosubview<memref<256xi32>>
            %outputSubview = aie.objectfifo.acquire @of_out (Produce, 1) : !aie.objectfifosubview<memref<256xi32>>

            // Access the first, and only, element of each subview.
            %input = aie.objectfifo.subview.access %inputSubview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
            %output = aie.objectfifo.subview.access %outputSubview[0] : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>

            %idx1 = arith.constant 3 : index
            %d1   = memref.load %input[%idx1] : memref<256xi32>
            %c1   = arith.constant 100 : i32
            %d2   = arith.addi %d1, %c1 : i32
            %idx2 = arith.constant 5 : index
            memref.store %d2, %input[%idx2] : memref<256xi32>

            memref.store %d1, %output[%idx1] : memref<256xi32>
            memref.store %d2, %output[%idx2] : memref<256xi32>

            // Release the previously acquired objects.
            // This is equivalent to releasing an AIE lock after accessing an AIE buffer.
            // This core releases objects both as a Consumer of one object FIFO and as a Producer of another:
            // this impacts the release values of the locks that are generated through the object FIFO lowering.
            aie.objectfifo.release @of_in (Consume, 1)
            aie.objectfifo.release @of_out (Produce, 1)
            aie.end
        }
    }
}
