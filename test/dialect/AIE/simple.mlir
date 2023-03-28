//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK: %[[T01:.*]] = AIE.tile(0, 1)
// CHECK: %[[T12:.*]] = AIE.tile(1, 2)
// CHECK: AIE.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

module {
  %01 = AIE.tile(0, 1)
  %12 = AIE.tile(1, 2)
  %02 = AIE.tile(49, 8) // Largest valid indices
  AIE.flow(%01, DMA : 0, %12, Core : 1)
}
