// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A stale reservation must not grant the linker memory outside this tile.
// RUN: not aie-translate --tilecol=0 --tilerow=2 --aie-generate-ldscript %s 2>&1 | FileCheck %s
// CHECK: data region runs past this tile's local memory; the buffer allocator's placement is stale

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 2)
    %data = aie.buffer(%t) { sym_name = "core_data", address = 65024 : i32, core_data } : memref<1024xi8>
    %c = aie.core(%t) { aie.end } { stack_size = 1024 : i32 }
  }
}
