//===- cpp_link_with_no_link_with.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Verify that aie-assign-core-link-files is a no-op on designs that carry
// no link_with attributes anywhere — no link_files attribute should appear
// on any CoreOp, and no warnings should be emitted.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | FileCheck %s

// CHECK-NOT: link_files
// CHECK-NOT: link_with

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }
  }
}
