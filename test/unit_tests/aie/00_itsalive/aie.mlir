//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc --unified %s
// RUN: %aiecc --no-unified %s

module @test00_itsalive {
aie.device(xcvc1902) {

  %tile12 = aie.tile(1, 2)

  %buf12_0 = aie.buffer(%tile12) { sym_name = "a" } : memref<256xi32>

  %core12 = aie.core(%tile12) {
    %val1 = arith.constant 1 : i32
    %idx1 = arith.constant 3 : index
    %2 = arith.addi %val1, %val1 : i32
    aie.end
  }

}
}
