//===- cpp_link_with_mode_verifier.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// CoreOp verifier rules for the two canonical link lists:
//   * a path may not appear in both 'link_files' and 'link_merge_files';
//   * the deprecated core-level 'link_with' conflicts with either list.
// The deprecated attribute has no way to request merging -- it always means an
// ordinary final-link input.

// RUN: aie-opt --verify-diagnostics --split-input-file %s

// The same artifact cannot be both merged and object-linked by one core.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-error@+1 {{artifact 'dup.o' appears in both 'link_files' and 'link_merge_files'}}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_files = ["a.o", "dup.o"], link_merge_files = ["dup.o", "b.o"]}
  }
}

// -----

// Deprecated core-level link_with conflicts with link_files.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-error@+1 {{cannot specify both 'link_with' (deprecated) and 'link_files'}}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "a.o", link_files = ["b.o"]}
  }
}

// -----

// Deprecated core-level link_with conflicts with link_merge_files too.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-error@+1 {{cannot specify both 'link_with' (deprecated) and 'link_merge_files'}}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "a.o", link_merge_files = ["b.ll"]}
  }
}

// -----

// Disjoint lists coexist happily; suffixes are irrelevant to the verifier.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_files = ["linked.bc"], link_merge_files = ["merged.o"]}
  }
}
