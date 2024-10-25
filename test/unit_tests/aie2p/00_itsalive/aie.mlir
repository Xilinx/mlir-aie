//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// REQUIRES: aiesimulator

// RUN: %python aiecc.py --no-compile --aiesim %s %S/test.cpp
// RUN: aie.mlir.prj/aiesim.sh | FileCheck %s

// CHECK: AIE2p ISS
// CHECK: Hello, world.
// CHECK: Exiting!

module @test00_itsalive {
  aie.device(npu2) {
    %tile20 = aie.tile(2, 0)
    %tile22 = aie.tile(2, 2)

    aie.flow(%tile20, DMA : 0, %tile22, DMA : 0)
    aie.flow(%tile22, DMA : 0, %tile20, DMA : 0)
  }
}