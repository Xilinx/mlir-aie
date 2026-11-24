//===- nested-module-device.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A module whose devices sit in nested modules has its devices already: the
// pass leaves it as it is rather than wrap the nested modules, devices and
// flows into a default xcvc1902 device, where the router aborted on the
// inner devices' flows.

// RUN: aie-opt --aie-canonicalize-device %s | FileCheck %s
// RUN: aie-opt --pass-pipeline='builtin.module(aie-resource-stages)' %s | FileCheck %s

// CHECK-NOT: xcvc1902
// CHECK: module @inner
// CHECK: aie.device(npu1_1col)

module {
  module @inner {
    aie.device(npu1_1col) {
      %t = aie.tile(0, 2)
      %s = aie.tile(0, 0)
      aie.flow(%t, DMA : 0, %s, DMA : 0)
    }
  }
}
