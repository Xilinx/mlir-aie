//===- cpp_link_with_mode_bad_metadata.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// aie-assign-core-link-files rejects malformed link_with_mode metadata:
//   * link_with_mode with no link_with to describe, and
//   * a link_with_mode value other than "merge".

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-assign-core-link-files %s

// link_with_mode without link_with has nothing to describe.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-error@+1 {{func 'no_artifact' has link_with_mode but no link_with}}
    func.func private @no_artifact() attributes {link_with_mode = "merge"}

    %core_0_2 = aie.core(%tile_0_2) {
      func.call @no_artifact() : () -> ()
      aie.end
    }
  }
}

// -----

// "merge" is currently the only accepted mode.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // expected-error@+1 {{func 'bad_mode' has unknown link_with_mode 'inline'; the only supported value is 'merge'}}
    func.func private @bad_mode() attributes {link_with = "k.o", link_with_mode = "inline"}

    %core_0_2 = aie.core(%tile_0_2) {
      func.call @bad_mode() : () -> ()
      aie.end
    }
  }
}
