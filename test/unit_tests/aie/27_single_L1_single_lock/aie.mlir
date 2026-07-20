//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON aiecc.py %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags %extraAieCcFlags% -o test.elf -- %S/test.cpp
// RUN: %run_on_vck5000 ./test.elf

module @test27_simple_shim_dma_single_lock {
  aie.device(xcvc1902) {
    %tile72 = aie.tile(7, 2)
    %lockCore = aie.lock(%tile72, 0)  {init = 0 : i32 , sym_name = "coreLock"} //{ init = 0 : i32 , sym_name = "coreLock"}
    %dummyLock = aie.lock(%tile72, 1) { sym_name = "dummyLock"}
    %buf72_0 = aie.buffer(%tile72) {sym_name = "aieL1" } : memref<16xi32>

    %core72 = aie.core(%tile72) {
      %c0 = arith.constant 0 : index

      %constant0 = arith.constant 0 : i32
      %constant7 = arith.constant 7 : i32
      %constant13 = arith.constant 13 : i32
      %constant43 = arith.constant 43 : i32
      %constant47 = arith.constant 47 : i32

      %c0_ul0 = arith.constant 0 : i32
      aie.use_lock(%lockCore, "Acquire", %c0_ul0)
      memref.store %constant7, %buf72_0[%c0] : memref<16xi32>
      %c1_ul1 = arith.constant 1 : i32
      aie.use_lock(%lockCore, "Release", %c1_ul1)

      %c0_ul2 = arith.constant 0 : i32
      aie.use_lock(%lockCore, "Acquire", %c0_ul2)
      memref.store %constant13, %buf72_0[%c0] : memref<16xi32>
      %c1_ul3 = arith.constant 1 : i32
      aie.use_lock(%lockCore, "Release", %c1_ul3)

      %c0_ul4 = arith.constant 0 : i32
      aie.use_lock(%lockCore, "Acquire", %c0_ul4)
      memref.store %constant43, %buf72_0[%c0] : memref<16xi32>
      %c1_ul5 = arith.constant 1 : i32
      aie.use_lock(%lockCore, "Release", %c1_ul5)

      %c0_ul6 = arith.constant 0 : i32
      aie.use_lock(%lockCore, "Acquire", %c0_ul6)
      memref.store %constant47, %buf72_0[%c0] : memref<16xi32>
      %c1_ul7 = arith.constant 1 : i32
      aie.use_lock(%lockCore, "Release", %c1_ul7)

      aie.end
    }
  }
}
