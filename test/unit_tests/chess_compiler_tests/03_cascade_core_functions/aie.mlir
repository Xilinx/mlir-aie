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
// RUN: xchesscc -p me -P ${CARDANO}/data/cervino/lib -c %S/kernel.cc
// RUN: aiecc.py --sysroot=%VITIS_SYSROOT% %s -I%S/../../../../runtime_lib/ %S/../../../../runtime_lib/test_library.cpp %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module {
  %tile13 = AIE.tile(1, 3)
  %tile14 = AIE.tile(2, 3)

  %buf13_0 = AIE.buffer(%tile13) { sym_name = "a" } : memref<256xi32>
  %buf14_0 = AIE.buffer(%tile14) { sym_name = "c" } : memref<256xi32>

  %lock13_3 = AIE.lock(%tile13, 3) // input buffer lock
  %lock14_7 = AIE.lock(%tile14, 7) // output buffer lock
  
  func private @do_mul(%A: memref<256xi32>) -> ()
  func private @do_mac(%A: memref<256xi32>) -> ()
  
  %core13 = AIE.core(%tile13) { 
    // AIE.useLock(%lock13_3, "Acquire", 1) // acquire for read(e.g. input ping)
    call @do_mul(%buf13_0) : (memref<256xi32>) -> ()
    // AIE.useLock(%lock13_3, "Release", 0) // release for write
    AIE.end
  } { link_with="kernel.o" }
  
  %core14 = AIE.core(%tile14) {
    %val1 = constant 7 : i32
    %idx1 = constant 0 : index
    memref.store %val1, %buf14_0[%idx1] : memref<256xi32>
    // AIE.useLock(%lock14_7, "Acquire", 0) // acquire for write
    call @do_mac(%buf14_0) : (memref<256xi32>) -> ()
    // AIE.useLock(%lock14_7, "Release", 1) // release for read
    AIE.end
  } { link_with="kernel.o" }
  
}
