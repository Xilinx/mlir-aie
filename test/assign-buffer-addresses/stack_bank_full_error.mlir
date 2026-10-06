// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The default allocator must not retry an unsatisfiable stack bank constraint.
// RUN: not aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// CHECK: requires a 1024-byte stack in bank 1, but no contiguous aligned space remains after address-pinned buffers
// CHECK-NOT: trying basic
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "b", address = 16384 : i32 } : memref<16384xi8>
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_bank = 1 : i32 }
  }
}
