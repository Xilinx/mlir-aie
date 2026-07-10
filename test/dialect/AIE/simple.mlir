//===- simple.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s
// CHECK: %[[T01:.*]] = aie.tile(0, 1)
// CHECK: %[[T12:.*]] = aie.tile(1, 2)
// CHECK: aie.flow(%[[T01]], DMA : 0, %[[T12]], Core : 1)

module {
  %01 = aie.tile(0, 1)
  %12 = aie.tile(1, 2)
  %02 = aie.tile(49, 8) // Largest valid indices
  aie.flow(%01, DMA : 0, %12, Core : 1)
}
