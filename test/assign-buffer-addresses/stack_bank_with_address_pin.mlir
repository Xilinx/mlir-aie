// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A stack pinned only to a bank can move around an immovable buffer in it.
// RUN: aie-opt --split-input-file --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s
// CHECK: aie.buffer
// CHECK-SAME: address = 16384 : i32
// CHECK: stack_address = 17408 : i32
// CHECK-SAME: stack_bank = 1 : i32
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "b", address = 16384 : i32 } : memref<1024xi8>
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_bank = 1 : i32 }
  }

  // -----

  // AIE2P's 64-byte stack alignment is stricter than its 32-byte load/store bus.
  // The small pinned buffer leaves the first bus-aligned candidate at 0x4020,
  // but the stack must begin at 0x4040.
  // CHECK: aie.buffer
  // CHECK-SAME: address = 16384 : i32
  // CHECK: stack_address = 16448 : i32
  // CHECK-SAME: stack_bank = 1 : i32
  module {
    aie.device(npu2) {
      %t = aie.tile(0, 2)
      %b = aie.buffer(%t) { sym_name = "b", address = 16384 : i32 } : memref<32xi8>
      %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_bank = 1 : i32 }
    }
  }
}
