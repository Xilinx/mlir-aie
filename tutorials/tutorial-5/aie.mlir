//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// aiecc.py -j4 --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%aie_runtime_lib%/  %extraAieCcFlags% %aie_runtime_lib%/test_library.cpp %S/test.cpp -o tutorial-5.exe

// REQUIRES: valid_xchess_license
// RUN: make -C %S
// RUN: %run_on_board ./tutorial-5.exe
// RUN: make -C %S clean

// Declare this MLIR module. A wrapper that can contain all 
// AIE tiles, buffers, and data movement
module @tutorial_5 {
    AIE.device(xcvc1902) {
        // 1 tile in row 4 (col 3)
        // even rows have local memory to its left
        %tile34 = AIE.tile(3, 4)

        // 1 tile in row 0 (col 7)
        // col 7, row 0 has access to a shimDMA
        %tile70 = AIE.tile(7, 0)

        // Declare external buffers, which represent pointers to external memory locations.
        %ext_buf70_in  = AIE.external_buffer {sym_name = "ddr_test_buffer_in"}: memref<256xi32> 
        %ext_buf70_out = AIE.external_buffer {sym_name = "ddr_test_buffer_out"}: memref<256xi32> 

        // Declare an object FIFO between the producer shim tile (7,0) and consumer tile (3,4).
        // The size of the object FIFO, i.e. its number of elements, is 1.
        // Objects, i.e. allocated memory elements, have type memref<256xi32>.
        // These tiles do not share memory between them.
        %objFifo_in = AIE.objectFifo.createObjectFifo(%tile70, {%tile34}, 1) : !AIE.objectFifo<memref<256xi32>>

        // Declare an object FIFO between the producer tile (3,4) and consumer shim tile (7,0).
        %objFifo_out = AIE.objectFifo.createObjectFifo(%tile34, {%tile70}, 1) : !AIE.objectFifo<memref<256xi32>>

        // Register the external memory pointers to the object FIFOs.
        AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo_in : !AIE.objectFifo<memref<256xi32>>, {%ext_buf70_in}) : (memref<256xi32>)
        AIE.objectFifo.registerExternalBuffers(%tile70, %objFifo_out : !AIE.objectFifo<memref<256xi32>>, {%ext_buf70_out}) : (memref<256xi32>)
    
        // Define core algorithm for tile(3,4) which reads value set by tile(1,4)
        // buf[5] = buf[3] + 100
        %core34 = AIE.core(%tile34) {
            // Acquire a subview with one object from each object FIFO.
            // This is equivalent to acquiring an AIE lock before accessing an AIE buffer.
            // This core acquires objects both as a Consumer of one object FIFO and as a Producer of another: 
            // this impacts the acquire values of the locks that are generated through the object FIFO lowering
            %inputSubview = AIE.objectFifo.acquire<Consume>(%objFifo_in : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
            %outputSubview = AIE.objectFifo.acquire<Produce>(%objFifo_out : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
            
            // Access the first, and only, element of each subview.
            %input = AIE.objectFifo.subview.access %inputSubview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
            %output = AIE.objectFifo.subview.access %outputSubview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

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
            AIE.objectFifo.release<Consume>(%objFifo_in : !AIE.objectFifo<memref<256xi32>>, 1)
            AIE.objectFifo.release<Produce>(%objFifo_out : !AIE.objectFifo<memref<256xi32>>, 1)
            AIE.end
        } 
    }
}
