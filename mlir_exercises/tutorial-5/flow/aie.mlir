//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%host_runtime_lib%/test_lib/include %extraAieCcFlags% -L%host_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o tutorial-5.exe
// RUN: %run_on_board ./tutorial-5.exe


// Declare this MLIR module. A wrapper that can contain all 
// AIE tiles, buffers, and data movement
module @tutorial_5 {

    // 1 tile in row 4 (col 3)
    // even rows have local memory to its left
    %tile34 = aie.tile(3, 4)

    // 1 tile in row 0 (col 7)
    // col 7, row 0 has access to a shim_dma
    %tile70 = aie.tile(7, 0)

    // Declare local memory of tile(1,4) and tile (3,4) which are not shared
    %buf34 = aie.buffer(%tile34) { sym_name = "a34" } : memref<256xi32>

    // Declare external buffers, which represent pointers to external memory locations.
    %ext_buf70_in  = aie.external_buffer {sym_name = "ddr_test_buffer_in"}: memref<256xi32>
    %ext_buf70_out = aie.external_buffer {sym_name = "ddr_test_buffer_out"}: memref<256xi32>

    // Declare local locks for tile(3,4) and shim tile (7,0) giving new
    // unique lock ID values 7 and 8
    %lock34_in  = aie.lock(%tile34, 7) { sym_name = "tile34_in_lock" }
    %lock34_out = aie.lock(%tile34, 8) { sym_name = "tile34_out_lock" }
    %lock70_in  = aie.lock(%tile70, 3) { sym_name = "ddr_test_buffer_in_lock" }
    %lock70_out = aie.lock(%tile70, 4) { sym_name = "ddr_test_buffer_out_lock" }

    // Connect DMA channel 0 on tile(7,0) to DMA channel 1 in tile(3,4)
    // with automatic shortest distance routing
    aie.flow(%tile70, DMA: 0, %tile34, DMA: 1)
    aie.flow(%tile34, DMA: 0, %tile70, DMA: 0)

    // shim DMA programming is nearly identical to tile DMA programming
    // shim_dma are blocking on release 1 (user intervention)
    %shimdma70 = aie.shim_dma(%tile70) {
        aie.dma_start("MM2S", 0, ^bd1, ^ch2)
        ^ch2:
            aie.dma_start("S2MM", 0, ^bd2, ^end)
        ^bd1:
            // Lock used to allow host to start transfer
            aie.use_lock(%lock70_in, "Acquire", 1)
            aie.dma_bd(%ext_buf70_in : memref<256xi32>, 0, 256)
            aie.use_lock(%lock70_in, "Release", 0)
            aie.next_bd ^end
        ^bd2:
            aie.use_lock(%lock70_out, "Acquire", 1)
            aie.dma_bd(%ext_buf70_out : memref<256xi32>, 0, 256)
            aie.use_lock(%lock70_out, "Release", 0)
            aie.next_bd ^end
        ^end:
            aie.end
    }
 
    // Define core algorithm for tile(3,4) which reads value set by tile(1,4)
    // buf[5] = buf[3] + 100
    %core34 = aie.core(%tile34) {
        // This acquire will stall since locks are initialized to Release, 0
        aie.use_lock(%lock34_out, "Acquire", 0) // Acquire out lock
        aie.use_lock(%lock34_in, "Acquire", 1)  // Acquire in lock
            // This will block while tileDMA moves data so we want to acquire this 2nd
        %idx1 = arith.constant 3 : index
        %d1   = memref.load %buf34[%idx1] : memref<256xi32>
        %c1   = arith.constant 100 : i32 
        %d2   = arith.addi %d1, %c1 : i32
		%idx2 = arith.constant 5 : index
		memref.store %d2, %buf34[%idx2] : memref<256xi32> 

        // This release doesn't do much in our example but mimics ping-pong
        aie.use_lock(%lock34_in, "Release", 0) // Release in lock
        aie.use_lock(%lock34_out, "Release", 1) // Release out lock
        aie.end
    }


    // Define local tile memory behavior (i.e. tileDMA)
    %mem34 = aie.mem(%tile34) {
        // sequence of DMAs declaration and buffer descriptors (bd)
        // ^bd0 - first label/ bd definition to set
        // ^end - next label/ bd definition to set 
        // (here, that is aie.end to indicate no more)
        aie.dma_start("S2MM", 1, ^bd0, ^ch2)
        ^ch2:
             aie.dma_start("MM2S", 0, ^bd1, ^end)
        ^bd0:
            // Add locks behvaior around bd definition
            aie.use_lock(%lock34_in, "Acquire", 0)
            // bd definition
            // %buf34 - local buffer
            // 0   - offset of transfer
            // 256 - length of transfer
            // 0   - A/B mode enable (default is disabled)
            aie.dma_bd(%buf34 : memref<256xi32>, 0, 256)
            aie.use_lock(%lock34_in, "Release", 1)
            aie.next_bd ^end
        ^bd1:
            aie.use_lock(%lock34_out, "Acquire", 1)
            aie.dma_bd(%buf34 : memref<256xi32>, 0, 256)
            aie.use_lock(%lock34_out, "Release", 0)
            aie.next_bd ^end
        ^end:
            aie.end
    }    

}
