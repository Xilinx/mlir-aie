//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: valid_xchess_license
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%aie_runtime_lib%/ %aie_runtime_lib%/test_library.cpp %S/test.cpp -o tutorial-1.elf
// RUN: %run_on_board ./tutorial-1.elf


// Declare this MLIR module. A wrapper that can contain all 
// AIE tiles, buffers, and data movement
module @tutorial_1 {

    // Declare tile in position col 1, row 4
    %tile14 = AIE.tile(1, 4)

    // Declare buffer for tile(1, 4) with symbolic name "a14" and 
    // size 256 deep x int32 wide. By default, the address of 
    // this buffer begins after the stack (1024 Bytes offset) and 
    // all subsequent buffers are placed immediately following thee
    // previous buffer in memory.
    %buf = AIE.buffer(%tile14) { sym_name = "a14" } : memref<256xi32>

    // declare kernel function name "extern_kernel" with one positional 
    // function argument, in this case mapped to a memref
    func.func private @extern_kernel(%b: memref<256xi32>) -> ()

    // Define core algorithm for tile(1, 4)
    // buf[3] = 14
    %core14 = AIE.core(%tile14) {
        // Call function and map local buffer %buf to function argument
        func.call @extern_kernel(%buf) : (memref<256xi32>) -> ()
        AIE.end
    } { link_with="kernel.o" } // indicate kernel object name used by this core

}
