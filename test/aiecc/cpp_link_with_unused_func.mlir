//===- cpp_link_with_unused_func.mlir ---------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test that a func.func carrying link_with that is never called from any core
// produces a warning from aie-assign-core-link-files.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-warning@+1 {{func 'never_called' has link_with but is never called from any core; its .o file will not be linked}}
    func.func private @never_called(memref<16xi32>) attributes {link_with = "x.o"}

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }
  }
}
