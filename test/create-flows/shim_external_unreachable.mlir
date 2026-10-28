//===- shim_external_unreachable.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-create-pathfinder-flows %s 2>&1 | FileCheck %s
// CHECK: error: Unable to find a legal routing: no path leads from (1, 0) DMA:0 to (1, 0) NOC:4

module {
  aie.device(npu1) {
    %t10 = aie.tile(1, 0)
    aie.flow(%t10, DMA : 0, %t10, NOC : 4)
  }
}