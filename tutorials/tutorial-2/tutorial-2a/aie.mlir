//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// aiecc.py -j4 -%VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o tutorial-2a.exe

// REQUIRES: valid_xchess_license
// RUN: make -C %S
// RUN: %run_on_board ./tutorial-2a.exe
// RUN: make -C %S clean


// Declare this MLIR module. A block encapsulates all
// AIE tiles, buffers, and communication in an AI Engine design
module @tutorial_2a {

    // Declare tile object of the AIE class located at position col 1, row 4
    %tile14 = aie.tile(1, 4)

    // Declare buffer for tile(1, 4) with symbolic name "a14" and
    // size 256 deep x int32 wide. By default, the address of
    // this buffer begins after the stack (1024 Bytes offset) and
    // all subsequent buffers are allocated one after another in memory.
    %buf = aie.buffer(%tile14) { sym_name = "a14" } : memref<256xi32>

    // Declare a lock 0 associated with tile(1,4) with a
    // symbolic name "lock14_0" which can be used by access functions
    // in the generated host API (aie.mlir.prj/aie_inc.cpp)
    %lock14_0 = aie.lock(%tile14, 0) { sym_name = "lock14_0" }

    // Define the algorithm for the core of tile(1, 4)
    // buf[3] = 14
    %core14 = aie.core(%tile14) {
        // Acquire lock right when core starts
        aie.use_lock(%lock14_0, "Acquire", 0)

		%val = arith.constant 14 : i32 // declare a constant (int32)
		%idx = arith.constant 3 : index // declare a constant (index)
		memref.store %val, %buf[%idx] : memref<256xi32> // store val in buf[3]

        // Release acquired lock at end of program.
        // This can be used by host to mark beginning/end of a program or
        // when the host is trying to determine when the program is done
        // by acquiring this lock (with value 1).
        aie.use_lock(%lock14_0, "Release", 1)
        aie.end
    }

}
