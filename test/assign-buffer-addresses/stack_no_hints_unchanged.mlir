//===- stack_no_hints_unchanged.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A core that asks for nothing keeps the layout it had before the stack could
// move: the stack at offset 0 and the buffers packed above it. The placement
// attributes stay absent rather than being materialised.

// RUN: aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s
// RUN: aie-opt --aie-assign-buffer-addresses %s 2>&1 | FileCheck %s

// CHECK: aie.buffer
// CHECK-SAME: address = 1024 : i32
// CHECK: aie.core
// CHECK-NOT: stack_address
// CHECK-NOT: stack_bank

module @stack_no_hints_unchanged {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %b = aie.buffer(%t) { sym_name = "b" } : memref<64xi32>
    %c = aie.core(%t) {
      aie.end
    } { stack_size = 1024 : i32 }
  }
}
