//===- cpp_link_with_mode_device_conflict.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The merge/link decision for an artifact must be consistent across a whole
// aie.device, not merely within one core.  aiecc's "unified" flow builds a
// single LLVM module for every core of a device and llvm-links the merge set
// into it once; if core A merges k.a while core B object-links it, core B's ELF
// would define the artifact's symbols twice.  aie-assign-core-link-files runs
// on the DeviceOp, so it has the scope to catch this.

// RUN: aie-opt --verify-diagnostics --split-input-file --aie-assign-core-link-files %s

// Two cores of one device disagree about the same artifact.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    func.func private @merged() attributes {link_with = "shared.a", link_with_mode = "merge"}
    func.func private @linked() attributes {link_with = "shared.a"}

    // expected-error@+1 {{artifact 'shared.a' is merged into an LLVM module here but object-linked in the same aie.device}}
    %core_0_2 = aie.core(%tile_0_2) {
      func.call @merged() : () -> ()
      aie.end
    }

    // expected-note@+1 {{artifact 'shared.a' is object-linked here}}
    %core_0_3 = aie.core(%tile_0_3) {
      func.call @linked() : () -> ()
      aie.end
    }
  }
}

// -----

// The same check also covers a single core that reaches one artifact through
// two declarations that disagree about the mode; the pass diagnoses it before
// writing the (invalid) attributes.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    func.func private @merged() attributes {link_with = "shared.a", link_with_mode = "merge"}
    func.func private @linked() attributes {link_with = "shared.a"}

    // expected-error@+2 {{artifact 'shared.a' is merged into an LLVM module here but object-linked in the same aie.device}}
    // expected-note@+1 {{artifact 'shared.a' is object-linked here}}
    %core_0_2 = aie.core(%tile_0_2) {
      func.call @merged() : () -> ()
      func.call @linked() : () -> ()
      aie.end
    }
  }
}

// -----

// Different artifacts with different modes are fine, even when the file
// suffixes suggest otherwise.
module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    func.func private @merged() attributes {link_with = "merged.o", link_with_mode = "merge"}
    func.func private @linked() attributes {link_with = "linked.bc"}

    %core_0_2 = aie.core(%tile_0_2) {
      func.call @merged() : () -> ()
      aie.end
    }

    %core_0_3 = aie.core(%tile_0_3) {
      func.call @linked() : () -> ()
      aie.end
    }
  }
}
