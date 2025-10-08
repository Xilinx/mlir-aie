//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// aiecc.py -j4 -%VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s -I%aie_runtime_lib%/test_lib/include %extraAieCcFlags% -L%aie_runtime_lib%/test_lib/lib -ltest_lib %S/test.cpp -o tutorial-8.exe

// REQUIRES: valid_xchess_license
// RUN: make -f %S/Makefile
// RUN: %run_on_board ./tutorial-8.exe
// RUN: make -f %S/Makefile clean

// Declare this MLIR module. A wrapper that can contain all
// AIE tiles, buffers, and data movement
module @tutorial_8 {

    // 2 tiles in row 3 (col 1 and col 2)
    // odd rows have local memory to its right
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)

    // Declare local memory of tile(2,4) which is shared with tile(1,4)
    %buf = aie.buffer(%tile23) { sym_name = "a23" } : memref<256xi32>

    // declare 2 kernel functions name "extern_kernel1" and "extern_kernel2"
    // with one positional function argument, in this case mapped to a memref
    func.func private @extern_kernel1() -> ()
    func.func private @extern_kernel2(%b: memref<256xi32>) -> ()

    // Declare shared lock (belonging to tile(2,4), lock ID=1)
    // %lock13_1 = aie.lock(%tile13, 1) { sym_name = "lock_13_1" }
    %lock13_2 = aie.lock(%tile13, 2) { sym_name = "lock_13_2" }

    // Define core algorithm for tile(1,4)
    // buf[3] = 13
    %core13 = aie.core(%tile13) {
        // Locks init value is Release 0, so this will always succeed first
        // aie.use_lock(%lock13_1, "Acquire", 0)

		// %val = arith.constant 14 : i384
        // aie.putCascade(%val : i384)

        func.call @extern_kernel1() : () -> ()

        // aie.use_lock(%lock13_1, "Release", 1)
        aie.end
    } { link_with="kernel1.o" }

    // Define core algorithm for tile(2,4) which reads value set by tile(1,4)
    // buf[5] = buf[3] + 100
    %core23 = aie.core(%tile23) {
        // This acquire succeeds when the core is enabled
        aie.use_lock(%lock13_2, "Acquire", 0)

        // This acquire will stall since locks are initialized to Release, 0
        // aie.use_lock(%lock13_1, "Acquire", 1)

        // %cas1 = aie.get_cascade() : i384
        // %d1   = arith.trunci %cas1 : i384 to i32
        // %c1   = arith.constant 100 : i32
        // %d2   = arith.addi %d1, %c1 : i32
		// %idx2 = arith.constant 5 : index
		// memref.store %d2, %buf[%idx2] : memref<256xi32>

        func.call @extern_kernel2(%buf) : (memref<256xi32>) -> ()

        // aie.use_lock(%lock13_1, "Release", 0)

        // This release means our 2nd core is done
        aie.use_lock(%lock13_2, "Release", 1)
        aie.end
    } { link_with="kernel2.o" }

}
