// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Merged kernels are compiled with the core's selected stack address space.
// No kernel file is needed when requesting only this graph node.
// RUN: %aiecc --get='perCoreStackSpace_{0}.txt' --tmpdir=%t.prj --output-dir=%t.out %s
// RUN: FileCheck %s --input-file=%t.out/perCoreStackSpace_main_core_0_2.txt
// RUN: not %aiecc --xchesscc --get='perCoreStackSpace_{0}.txt' --tmpdir=%t.prj %s 2>&1 | FileCheck %s --check-prefix=CHESS
// CHECK: 6
// CHESS: a stack outside memory bank A requires Peano compilation

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    func.func private @kernel() attributes {link_with = "kernel.ll", link_with_mode = "merge"}
    %core = aie.core(%tile) {
      func.call @kernel() : () -> ()
      aie.end
    } {stack_bank = 1 : i32, stack_size = 1024 : i32}
  }
}
