//===- cpp_link_with_both_attrs.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that a core with both the deprecated 'link_with' scalar attr AND the
// canonical 'link_files' array attr on the same CoreOp is rejected by the
// CoreOp verifier.

// RUN: aie-opt --verify-diagnostics %s

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-error@+1 {{cannot specify both 'link_with' (deprecated) and 'link_files'}}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "a.o", link_files = ["b.o"]}
  }
}
