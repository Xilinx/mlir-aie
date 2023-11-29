//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// aiecc.py -j4 --sysroot=%VITIS_SYSROOT% --host-target=aarch64-linux-gnu %s -I%host_runtime_lib%/  %extraAieCcFlags% %host_runtime_lib%/test_library.cpp %S/test.cpp -o tutorial-7.exe

// REQUIRES: valid_xchess_license
// RUN: make -C %S
// RUN: %run_on_board ./tutorial-7.exe
// RUN: make -C %S clean

// Declare this MLIR module. A wrapper that can contain all 
// AIE tiles, buffers, and data movement
module @tutorial_7 {

    AIE.device(xcvc1902) {
        // 2 tiles in row 4 (col 1 and col 3) and 1 in row 5 (col 3)
        // even rows have local memory to its left
        // odd rows have local memory to its right
        %tile14 = AIE.tile(1, 4) 
        %tile34 = AIE.tile(3, 4)
        %tile35 = AIE.tile(3, 5)

        // Declare an object FIFO between the producer shim tile (7,0) and consumer tiles (3,4) and (3,5).
        // As there are multiple consumers, this objectFifo represents a one-to-many broadcast.
        // The size of the object FIFO, i.e. its number of elements, is 1.
        // Objects, i.e. allocated memory elements, have type memref<256xi32>.
        // Each (producer tile / consumer tile) pair does not share memory.
        AIE.objectfifo @of (%tile14, {%tile34,%tile35}, 1 : i32) : !AIE.objectfifo<memref<256xi32>>

        // These locks will be used to gate when our end cores are done
        %lock34_8 = AIE.lock(%tile34, 8) { sym_name = "lock_a34_8" }
        %lock35_8 = AIE.lock(%tile35, 8) { sym_name = "lock_a35_8" }

        // Define core algorithm for tile(1,4)
        // buf[3] = 14
        %core14 = AIE.core(%tile14) {
            // Acquire a subview with one object from the object FIFO.
            // This is equivalent to acquiring an AIE lock before accessing an AIE buffer.
            // This core acquires objects as a Producer: this impacts the acquire value of the lock 
            // that is generated through the object FIFO lowering.
            %inputSubview = AIE.objectfifo.acquire @of (Produce, 1) : !AIE.objectfifosubivew<memref<256xi32>>
            
            // Access the first, and only, element of the subview.
            %input = AIE.objectfifo.subview.access %inputSubview[0] : !AIE.objectfifosubivew<memref<256xi32>> -> memref<256xi32>

            %val = arith.constant 14 : i32 
            %idx = arith.constant 3 : index 
            memref.store %val, %input[%idx] : memref<256xi32> 
            
            // Release the previously acquired object.
            // This is equivalent to releasing an AIE lock after accessing an AIE buffer.
            // This core releases objects as a Producer: this impacts the release value of the lock 
            // that is generated through the object FIFO lowering.
            AIE.objectfifo.release @of (Produce, 1)
            AIE.end
        } 

        // Define core algorithm for tile(3,4) which reads value set by tile(1,4)
        // buf[5] = buf[3] + 100
        %core34 = AIE.core(%tile34) {
            // This acquire succeeds when the core is enabled
            AIE.useLock(%lock34_8, "Acquire", 0)

            %inputSubview = AIE.objectfifo.acquire @of (Consume, 1) : !AIE.objectfifosubivew<memref<256xi32>>
            %input = AIE.objectfifo.subview.access %inputSubview[0] : !AIE.objectfifosubivew<memref<256xi32>> -> memref<256xi32>

            %idx1 = arith.constant 3 : index
            %d1   = memref.load %input[%idx1] : memref<256xi32>
            %c1   = arith.constant 100 : i32 
            %d2   = arith.addi %d1, %c1 : i32
            %idx2 = arith.constant 5 : index
            memref.store %d2, %input[%idx2] : memref<256xi32> 
            
            AIE.objectfifo.release @of (Consume, 1)

            // This release means our 2nd core is done
            AIE.useLock(%lock34_8, "Release", 1)
            AIE.end
        }

        // Define core algorithm for tile(3,5) which reads value set by tile(1,4)
        // buf[5] = buf[3] + 100
        %core35 = AIE.core(%tile35) {
            // This acquire succeeds when the core is enabled
            AIE.useLock(%lock35_8, "Acquire", 0)

            %inputSubview = AIE.objectfifo.acquire @of (Consume, 1) : !AIE.objectfifosubivew<memref<256xi32>>
            %input = AIE.objectfifo.subview.access %inputSubview[0] : !AIE.objectfifosubivew<memref<256xi32>> -> memref<256xi32>

            %idx1 = arith.constant 3 : index
            %d1   = memref.load %input[%idx1] : memref<256xi32>
            %c1   = arith.constant 100 : i32 
            %d2   = arith.addi %d1, %c1 : i32
            %idx2 = arith.constant 5 : index
            memref.store %d2, %input[%idx2] : memref<256xi32> 
            
            AIE.objectfifo.release @of (Consume, 1)

            // This release means our 2nd core is done
            AIE.useLock(%lock35_8, "Release", 1)
            AIE.end
        }
    }
}
