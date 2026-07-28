//===- cpp_link_with_mode_merge.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Link policy comes from metadata, never from the file name.  A func.func
// declaration carrying `link_with_mode = "merge"` routes its artifact into the
// core's `link_merge_files`; a declaration without `link_with_mode` routes its
// artifact into `link_files` no matter what the file is called.
//
// The file names here deliberately contradict the modes -- the merged artifact
// is named ".o" and the object-linked ones are named ".bc"/".ll" -- so that any
// reintroduction of extension sniffing fails this test.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | FileCheck %s

// Object-linked artifacts keep insertion order and land in link_files.
// CHECK-DAG: link_files = ["obj_a.bc", "obj_b.ll"]
// Merged artifacts land in link_merge_files, in insertion order.
// CHECK-DAG: link_merge_files = ["merge_a.o", "merge_b.o"]

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    // No link_with_mode: ordinary final-link inputs, despite the IR suffixes.
    func.func private @obj_a() attributes {link_with = "obj_a.bc"}
    func.func private @obj_b() attributes {link_with = "obj_b.ll"}
    // link_with_mode = "merge": merged into the core's LLVM module, despite
    // the object-file suffixes.
    func.func private @merge_a() attributes {link_with = "merge_a.o", link_with_mode = "merge"}
    func.func private @merge_b() attributes {link_with = "merge_b.o", link_with_mode = "merge"}
    // Duplicate call of an already-recorded artifact must not duplicate it.
    func.func private @merge_a_again() attributes {link_with = "merge_a.o", link_with_mode = "merge"}

    %core_0_2 = aie.core(%tile_0_2) {
      func.call @obj_a() : () -> ()
      func.call @merge_a() : () -> ()
      func.call @obj_b() : () -> ()
      func.call @merge_b() : () -> ()
      func.call @merge_a_again() : () -> ()
      aie.end
    }
  }
}
