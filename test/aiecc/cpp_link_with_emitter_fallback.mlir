//===- cpp_link_with_emitter_fallback.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test the deprecated fallback path in the ldscript and BCF emitters:
// when a core still has a core-level link_with (and no link_files), both
// emitters should still emit the correct entry without running
// aie-assign-core-link-files first.
//
// Only a func.func declaration can carry link_with_mode, so the core-level
// attribute has no way to request merging: it is always an ordinary link
// input, even when it names LLVM IR.  Core (0,3) pins that -- a `.ll` here
// must still reach the linker rather than being silently rerouted.

// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=LDSCRIPT
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=BCF
// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=3 %s | FileCheck %s --check-prefix=LDSCRIPT_IR
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=3 %s | FileCheck %s --check-prefix=BCF_IR

// LDSCRIPT: INPUT(fallback.o)
// BCF: _include _file fallback.o

// LDSCRIPT_IR: INPUT(fallback.ll)
// BCF_IR: _include _file fallback.ll

// Use bare cores without objectfifo so no lowering is needed before
// aie-translate can generate the ldscript/BCF.

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    // Core keeps a core-level link_with (no pass run, no link_files set).
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "fallback.o"}

    // Same, naming LLVM IR: the suffix does not change the routing.
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {link_with = "fallback.ll"}
  }
}
