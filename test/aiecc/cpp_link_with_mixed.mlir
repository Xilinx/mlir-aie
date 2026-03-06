//===- cpp_link_with_mixed.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Test that a core with both a deprecated core-level link_with AND a call to
// a func.func with its own link_with produces a merged, deduplicated link_files
// set.  The core-level attr is consumed (removed) and both .o paths appear
// exactly once in link_files.

// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | FileCheck %s --check-prefix=OPT
// RUN: aie-opt --verify-diagnostics --aie-assign-core-link-files %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s --check-prefix=LDSCRIPT

// The merged set must contain both files.
// OPT-DAG: "core_only.o"
// OPT-DAG: "func_only.o"
// The deprecated core-level attr must be gone.
// OPT-NOT: link_with = "core_only.o"

// LDSCRIPT-DAG: INPUT(core_only.o)
// LDSCRIPT-DAG: INPUT(func_only.o)

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    func.func private @ext(memref<16xi32>) attributes {link_with = "func_only.o"}

    // Core carries deprecated core-level link_with AND calls a func with its own.
    // expected-warning@+1 {{link_with on aie.core is deprecated; attach link_with to the func.func declaration instead}}
    %core_0_2 = aie.core(%tile_0_2) {
      %buf = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %buf[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      func.call @ext(%elem) : (memref<16xi32>) -> ()
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    } {link_with = "core_only.o"}

    aie.runtime_sequence() {}
  }
}
