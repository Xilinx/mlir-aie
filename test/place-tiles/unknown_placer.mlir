//===- unknown_placer.mlir --------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-place-tiles='placer=unknown_placer' %s 2>&1 | FileCheck %s

// Test: Unknown placer name should emit error
// CHECK: error: Unknown placer: unknown_placer
module @unknown_placer_error {
  aie.device(npu1) {
    %core = aie.logical_tile<CoreTile>(?, ?)
    aie.core(%core) { aie.end }
  }
}
