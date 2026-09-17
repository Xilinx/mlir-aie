//===- stack_bank_pin.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// stack_bank pins the stack the way mem_bank pins a buffer: the allocator
// picks an address inside that bank and records it, so every later reader sees
// one answer.
//
// Moving the stack off bank 0 is what lets a design keep a whole bank for
// something else, and what decides where an overrun lands.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s 2>&1 | FileCheck %s

// Bank 1 of an npu2 core tile is [0x4000, 0x8000).
// CHECK: stack_address = 16384 : i32
// CHECK-SAME: stack_bank = 1 : i32

module @stack_bank_pin {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "b" } : memref<1024xi32>
    %c = aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32, stack_bank = 1 : i32 }
  }
}
