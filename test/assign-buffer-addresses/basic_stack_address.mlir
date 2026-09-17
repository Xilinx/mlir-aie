// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Skipping a buffer below a relocated stack must not move the cursor backwards
// into the stack. A resolved stack bank is also legal in basic-sequential.
// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s

// CHECK: aie.buffer
// CHECK-SAME: address = 0 : i32
// CHECK: aie.buffer
// CHECK-SAME: address = 17408 : i32
// CHECK: stack_address = 16384 : i32
// CHECK-SAME: stack_bank = 1 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %low = aie.buffer(%t) { sym_name = "low", address = 0 : i32 } : memref<16384xi8>
    %b = aie.buffer(%t) { sym_name = "b" } : memref<1024xi8>
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 16384 : i32, stack_bank = 1 : i32 }
  }
}
