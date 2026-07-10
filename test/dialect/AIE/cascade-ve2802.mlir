//===- cascade-ve2802.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s

module @test {
  aie.device(xcve2802) {
    %t33 = aie.tile(3, 3)
    %c33 = aie.core(%t33) {
      %val2 = aie.get_cascade() : vector<16xi32>
      aie.put_cascade(%val2: vector<16xi32>)
      aie.end
 
    }
  }
}
