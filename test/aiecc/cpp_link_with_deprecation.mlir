//===- cpp_link_with_deprecation.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test that core-level link_with still compiles but emits a deprecation warning,
// and that the pass migrates the attribute to link_files on the core.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s
// RUN: aie-opt --aie-assign-core-link-files %s | FileCheck %s --check-prefix=MIGRATED

// Verify the pass migrated the deprecated core-level attr into link_files and
// removed link_with from the core.
// MIGRATED:     link_files = ["legacy.o"]
// MIGRATED-NOT: link_with = "legacy.o"

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // expected-warning@+1 {{link_with on aie.core is deprecated; attach link_with to the func.func declaration instead}}
    %core_0_2 = aie.core(%tile_0_2) {
      %buf = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    } {link_with = "legacy.o"}

    aie.runtime_sequence() {}
  }
}
