//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: !hsa
// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %s %test_lib_flags %extraAieCcFlags% %S/test.cpp -o test.elf
// RUN: %run_on_board ./test.elf

module @test27_simple_shim_dma_single_lock {
aie.device(xcvc1902) {

  %tile73 = aie.tile(7, 3)
  %lockCore = aie.lock(%tile73, 0) { sym_name = "coreLock"}
  %dummyLock = aie.lock(%tile73, 1) { sym_name = "dummyLock"}
  %buf73_0 = aie.buffer(%tile73) {sym_name = "aieL1" } : memref<16xi32>

  %core72 = aie.core(%tile73) {
    %c0 = arith.constant 0 : index

    %constant0 = arith.constant 0 : i32
    %constant7 = arith.constant 7 : i32
    %constant13 = arith.constant 13 : i32
    %constant43 = arith.constant 43 : i32
    %constant47 = arith.constant 47 : i32

    %c0_ul0 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul0)
    memref.store %constant7, %buf73_0[%c0] : memref<16xi32>
    %c1_ul1 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul1)

    %c0_ul2 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul2)
    %c1_ul3 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul3)
    // aie.use_lock(%dummyLock, "Acquire", 0)
    // aie.use_lock(%dummyLock, "Release", 0)

    %c0_ul4 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul4)
    memref.store %constant13, %buf73_0[%c0] : memref<16xi32>
    %c1_ul5 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul5)

    %c0_ul6 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul6)
    %c1_ul7 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul7)
    // aie.use_lock(%dummyLock, "Acquire", 0)
    // aie.use_lock(%dummyLock, "Release", 0)

    %c0_ul8 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul8)
    memref.store %constant43, %buf73_0[%c0] : memref<16xi32>
    %c1_ul9 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul9)

    %c0_ul10 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul10)
    %c1_ul11 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul11)
    // aie.use_lock(%dummyLock, "Acquire", 0)
    // aie.use_lock(%dummyLock, "Release", 0)

    %c0_ul12 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul12)
    memref.store %constant47, %buf73_0[%c0] : memref<16xi32>
    %c1_ul13 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul13)

    %c0_ul14 = arith.constant 0 : i32
    aie.use_lock(%lockCore, "Acquire", %c0_ul14)
    %c1_ul15 = arith.constant 1 : i32
    aie.use_lock(%lockCore, "Release", %c1_ul15)
    // aie.use_lock(%dummyLock, "Acquire", 0)
    // aie.use_lock(%dummyLock, "Release", 0)

    aie.end
  }
  
}
}
