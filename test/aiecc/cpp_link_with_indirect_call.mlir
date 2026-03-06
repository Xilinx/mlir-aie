//===- cpp_link_with_indirect_call.mlir -------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test that an indirect call inside a core body triggers a warning from
// aie-assign-core-link-files, since link_with on indirectly-called funcs
// cannot be statically resolved.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    func.func private @some_helper() -> ()

    %core_0_2 = aie.core(%tile_0_2) {
      %fptr = func.constant @some_helper : () -> ()
      // expected-warning@+1 {{indirect call in core body}}
      func.call_indirect %fptr() : () -> ()
      aie.end
    }
  }
}
