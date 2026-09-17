// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A per-tile override must reject unresolved stack pins just like the flag.
// RUN: not aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s 2>&1 | FileCheck %s
// CHECK: basic-sequential allocation cannot resolve stack_bank; use bank-aware allocation or specify stack_address

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2) { allocation_scheme = "basic-sequential" }
    %c = aie.core(%t) { aie.end } { stack_bank = 1 : i32 }
  }
}
