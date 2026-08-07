//===- peano_compat_merge_link.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Merging a kernel into the core (`link_with_mode = "merge"`) is the only step
// that hands *textual* LLVM IR between LLVM versions, so it is the only one
// exposed to the skew between aiecc's LLVM and Peano's older one.  The linker
// reprints the merged module in aiecc's dialect, and Peano then rejects its own
// kernel back unless that text is downgraded: `unterminated attribute group` on
// `nocreateundeforpoison`, `immarg operand has non-immediate parameter` on a
// size-less `llvm.lifetime.*`.
//
// The whole peano path runs here, so Peano's own opt/llc decide.  Drop the
// post-link downgrade and this fails at `opted_{0}.ll` with those errors.

// REQUIRES: peano

// RUN: rm -rf %t && mkdir -p %t
// RUN: aiecc --tmpdir %t %s
// RUN: FileCheck %s --check-prefix=LINKED --input-file %t/peano-linked_main_core_0_2.ll --implicit-check-not=nocreateundeforpoison --implicit-check-not=", align "
// RUN: FileCheck %s --check-prefix=OPTED --input-file %t/opted_main_core_0_2.ll --implicit-check-not=@merge_kernel

// The kernel arrived, and the merged text is back in a dialect Peano parses: no
// unknown attribute (--implicit-check-not above) and lifetime markers carrying
// the size operand its verifier requires.  `-1` is "whole object" -- the
// original size is gone, dropped when the newer LLVM read the kernel in.
// LINKED: define linkonce_odr void @merge_kernel
// LINKED-DAG: call void @llvm.lifetime.start.p0(i64 -1,
// LINKED-DAG: call void @llvm.lifetime.end.p0(i64 -1,
// LINKED-DAG: declare void @llvm.lifetime.start.p0(i64 immarg,
// LINKED-DAG: declare void @llvm.lifetime.end.p0(i64 immarg,

// Peano's opt then folds the alwaysinline kernel in and dead-strips the
// linkonce_odr definition: no `@merge_kernel` survives (--implicit-check-not
// above; the inliner's `merge_kernel.exit` label carries no `@`).
// OPTED: define void @core_0_2

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // Relative link_with resolves against the input file's directory, so the
    // artifact is referenced in place -- no copy into the work dir needed.
    func.func private @merge_kernel(memref<16xi32>, memref<16xi32>) attributes {link_with = "Inputs/peano_merge_intrinsics_kernel.ll", link_with_mode = "merge"}

    %core_0_2 = aie.core(%tile_0_2) {
      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @merge_kernel(%elem_in, %elem_out) : (memref<16xi32>, memref<16xi32>) -> ()
      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }
  }
}
