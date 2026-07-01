//===- test_placer_unknown.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles='placer=unknown_placer' %s 2>&1 | FileCheck %s

// Test: Unknown placer name should emit error
// CHECK: for the --placer option: Cannot find option named 'unknown_placer'!
module @unknown_placer_error {
  aie.device(npu1) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%core) { aie.end }
  }
}
