//===- cpp_link_with_emitter_fallback.mlir ---------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test the deprecated fallback path in the ldscript and BCF emitters:
// when a core still has a core-level link_with (and no link_files), both
// emitters should still emit the correct entry without running
// aie-assign-core-link-files first.

// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=LDSCRIPT
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=BCF

// LDSCRIPT: INPUT(fallback.o)
// BCF: _include _file fallback.o

// Use a bare core without objectfifo so no lowering is needed before
// aie-translate can generate the ldscript/BCF.

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // Core keeps the old core-level link_with (no pass run, no link_files set).
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "fallback.o"}
  }
}
