//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator, valid_xchess_license
// RUN: %aiecc --get-aiesim --xchesscc --xbridge %VitisSysrootFlag% --host-target=%aieHostTargetTriplet% %link_against_hsa% %s %test_lib_flags -o test.elf -- %S/test.cpp
// RUN: xchesscc_wrapper aie +l aie.mlir.prj/main_core_1_3.bcf %S/kernel.cc -o custom_1_3.elf
// RUN: %run_on_board ./test.elf
// RUN: ./aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: test start.
// CHECK: PASS!

module @test_chess_02_deprecated_precompiled_kernel {
aie.device(xcvc1902) {

  %tile13 = aie.tile(1, 3)

  aie.buffer @a(%tile13) : memref<256xi32>
  aie.buffer @b(%tile13) : memref<256xi32>

  aie.lock @input_lock(%tile13, 3)
  aie.lock @output_lock(%tile13, 5)

  %core13 = aie.core(%tile13) { aie.end } { elf_file = "custom_1_3.elf" }

}
}
