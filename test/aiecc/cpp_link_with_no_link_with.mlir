//===- cpp_link_with_no_link_with.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
