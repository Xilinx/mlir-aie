// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The bank used to schedule a precompiled kernel's stack accesses is unknown.
// Reject it before attempting compilation, for both lowering strategies.
// RUN: not %aiecc --get='perCoreStackSpace_{0}.txt' --tmpdir=%t.prj %s 2>&1 | FileCheck %s
// RUN: not %aiecc --get='perCoreStackSpace_{0}.txt' --tmpdir=%t.prj --no-unified %s 2>&1 | FileCheck %s
// RUN: sed 's/stack_bank = 1/stack_bank = 0/' %s > %t.mlir
// RUN: %aiecc --get='perCoreStackSpace_{0}.txt' --tmpdir=%t.prj --output-dir=%t.out %t.mlir
// RUN: FileCheck %s --check-prefix=DEFAULT --input-file=%t.out/perCoreStackSpace_main_core_0_2.txt
// DEFAULT: 5
// CHECK: a stack outside memory bank A requires Peano compilation with no separately compiled link_files
// CHECK-SAME: link_with_mode = "merge"

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    func.func private @kernel() attributes {link_with = "kernel.o"}
    %core = aie.core(%tile) {
      func.call @kernel() : () -> ()
      aie.end
    } {stack_bank = 1 : i32, stack_size = 1024 : i32}
  }
}
