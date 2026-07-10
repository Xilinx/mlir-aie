//===- badconnect.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt %s 2>&1 | FileCheck %s
// CHECK: error{{.*}} 'aie.connect' op source index cannot be less than zero

module {
  %20 = aie.tile(2, 0)
  aie.switchbox(%20) {
    aie.connect<East: -1, East: 0>
  }
}
