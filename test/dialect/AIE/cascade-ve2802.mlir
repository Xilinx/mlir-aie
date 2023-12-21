//===- cascade-ve2802.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s

module @test {
  aie.device(xcve2802) {
    %t33 = aie.tile(3, 3)
    %c33 = aie.core(%t33) {
      %val2 = aie.get_cascade() : vector<16xi32>
      aie.putCascade(%val2: vector<16xi32>)
      aie.end
 
    }
  }
}
