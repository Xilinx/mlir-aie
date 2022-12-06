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
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib%/ %aie_runtime_lib%/test_library.cpp %S/test.cpp -o tutorial-1.elf
// RUN: %run_on_board ./tutorial-1.elf


// Declare this MLIR module. A block encapsulates all 
// AIE tiles, buffers, and communication in an AI Engine design
module @tutorial_1 {

    // Declare tile object of the AIE class located at position col 1, row 4
    %tile14 = AIE.tile(1, 4)

    // Declare buffer for tile(1, 4) with symbolic name "a14" and 
    // size 256 deep x int32 wide. By default, the address of 
    // this buffer begins after the stack (1024 Bytes offset) and 
    // all subsequent buffers are allocated one after another in memory.
    %buf = AIE.buffer(%tile14) { sym_name = "a14" } : memref<256xi32>

    // Declare function signature of a function object (func.func) and
    // define the kernel function name "extern_kernel" with one positional 
    // function argument, memref that is 32b wide x 256 deep
    func.func private @extern_kernel(%b: memref<256xi32>) -> ()

    // Define the algorithm for the core of tile(1, 4) which is mapped
    // to an externally defined kernel function.
    // kernel function: buf[3] = 14
    %core14 = AIE.core(%tile14) {
        // Call kernel function and map buffer %buf to the function argument
        func.call @extern_kernel(%buf) : (memref<256xi32>) -> ()
        AIE.end
    } { link_with="kernel.o" } 
    // Link externally compiled object file used in this core

}
