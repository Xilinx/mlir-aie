//===- configure_cascade.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s

module @test {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    aie.configure_cascade(%t13, North, East)
    aie.configure_cascade(%t23, West, South)
  }
}
