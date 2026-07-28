//===- cpp_link_with_ir_artifacts_fallback.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Companion to cpp_link_with_ir_artifacts.mlir, covering the deprecated
// core-level link_with fallback (aie-assign-core-link-files not run, so no
// link_files array).  An LLVM IR artifact must be skipped on that path too --
// the emitters filter both branches, or a design that skipped the pass would
// hand the linker a .ll.
//
// As in cpp_link_with_emitter_fallback.mlir, a bare core is used so no
// lowering is needed before aie-translate.

// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=LDSCRIPT
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=BCF

// The ldscript/BCF still emit their usual scaffolding, just no entry for the
// IR artifact.
// LDSCRIPT-NOT: fallback.ll
// LDSCRIPT: PROVIDE(main = core_0_2)
// LDSCRIPT-NOT: fallback.ll

// BCF-NOT: fallback.ll
// BCF: _resolve _main core_0_2
// BCF-NOT: fallback.ll

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // Core keeps a core-level link_with naming an LLVM IR artifact.
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "fallback.ll"}
  }
}
