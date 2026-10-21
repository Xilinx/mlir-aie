// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Neither linker may silently place an unresolved bank pin at address zero.
// RUN: not aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s 2>&1 | FileCheck %s
// RUN: not aie-translate --tilecol=0 --tilerow=2 --aie-generate-bcf %s 2>&1 | FileCheck %s
// CHECK: stack_bank has no assigned stack_address; run --aie-assign-buffer-addresses with bank-aware allocation

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %c = aie.core(%t) { aie.end } { stack_bank = 1 : i32 }
  }
}
