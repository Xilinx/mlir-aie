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
// REQUIRES: peano
// RUN: %PYTHON aiecc.py --no-unified --xchesscc    --xbridge %s
// RUN: %PYTHON aiecc.py --unified    --xchesscc    --xbridge %s
// RUN: %PYTHON aiecc.py --no-unified --no-xchesscc --xbridge %s
// RUN: %PYTHON aiecc.py --unified    --no-xchesscc --xbridge %s
// RUN: %PYTHON aiecc.py --no-unified --xchesscc    --no-xbridge %s
// RUN: %PYTHON aiecc.py --unified    --xchesscc    --no-xbridge %s

module @test00_itsalive {
  %tile12 = aie.tile(1, 2)

  %buf12_0 = aie.buffer(%tile12) { sym_name = "a", address = 0 : i32 } : memref<256xi32>

  %core12 = aie.core(%tile12) {
    %val1 = arith.constant 1 : i32
    %idx1 = arith.constant 3 : index
    %2 = arith.addi %val1, %val1 : i32
    aie.end
  }
}
