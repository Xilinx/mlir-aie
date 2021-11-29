//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

module @benchmark14_timer {
  %t73 = AIE.tile(7, 3)


  %buf73_0 = AIE.buffer(%t73) { sym_name = "a" } : memref<256xi32>

  %core73 = AIE.core(%t73) {
    %val1 = constant 7 : i32
    %idx1 = constant 3 : index
    %2 = addi %val1, %val1 : i32
    memref.store %2, %buf73_0[%idx1] : memref<256xi32>

    AIE.end
  }

}