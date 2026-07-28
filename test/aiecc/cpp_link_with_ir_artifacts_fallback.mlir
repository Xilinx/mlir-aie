//===- cpp_link_with_ir_artifacts_fallback.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Companion to cpp_link_with_ir_artifacts.mlir, covering the deprecated
// core-level link_with.  Only a func.func declaration can carry
// link_with_mode, so the core-level attribute has nowhere to request merging:
// it is *always* an ordinary link input, even when it names a .ll or a .bc.
// Check both the emitters' deprecated fallback branch (no link_files, because
// aie-assign-core-link-files was not run) and the pass's migration of the
// attribute into link_files -- never link_merge_files.
//
// As in cpp_link_with_emitter_fallback.mlir, bare cores are used so no
// lowering is needed before aie-translate.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s
// RUN: aie-opt --aie-assign-core-link-files %s | FileCheck %s --check-prefix=OPT
// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=LD_LL
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 %s | FileCheck %s --check-prefix=BCF_LL
// RUN: aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=3 %s | FileCheck %s --check-prefix=LD_BC
// RUN: aie-translate --aie-generate-bcf --tilecol=0 --tilerow=3 %s | FileCheck %s --check-prefix=BCF_BC

// Migration keeps both artifacts on the ordinary path.
// OPT-NOT: link_merge_files
// OPT:     link_files = ["fallback.ll"]
// OPT-NOT: link_merge_files
// OPT:     link_files = ["fallback.bc"]
// OPT-NOT: link_merge_files

// The emitters' fallback branch hands the artifact to the linker unchanged.
// LD_LL:  INPUT(fallback.ll)
// LD_LL:  PROVIDE(main = core_0_2)
// BCF_LL: _include _file fallback.ll
// BCF_LL: _resolve _main core_0_2

// LD_BC:  INPUT(fallback.bc)
// LD_BC:  PROVIDE(main = core_0_3)
// BCF_BC: _include _file fallback.bc
// BCF_BC: _resolve _main core_0_3

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    // Cores keep a core-level link_with naming an LLVM IR artifact.
    // expected-warning@+1 {{link_with on aie.core is deprecated; attach link_with to the func.func declaration instead}}
    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    } {link_with = "fallback.ll"}

    // expected-warning@+1 {{link_with on aie.core is deprecated; attach link_with to the func.func declaration instead}}
    %core_0_3 = aie.core(%tile_0_3) {
      aie.end
    } {link_with = "fallback.bc"}
  }
}
