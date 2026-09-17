// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Moving the stack after placement invalidates a recorded data region.
// RUN: not aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s 2>&1 | FileCheck %s
// CHECK: data region overlaps the stack; the buffer allocator's placement is stale

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %data = aie.buffer(%t) { sym_name = "core_data", address = 16384 : i32, core_data } : memref<4096xi8>
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32, stack_address = 16384 : i32 }
  }
}
