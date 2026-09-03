//===- reserved_data_size_zero_size_buffer.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A zero-sized buffer pinned inside the free region must not fragment the
// ldscript `data` region: its length stays the full 64512 bytes (0xFC00) above
// the 1024-byte stack, matching the allocator's largestFreeRun().

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | aie-translate --aie-generate-ldscript --tilecol=0 --tilerow=2 | FileCheck %s

// CHECK: data (!RX) : ORIGIN = {{.*}}, LENGTH = 0xFC00

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %mid = aie.buffer(%tile_0_2) {address = 30016 : i32, sym_name = "mid"} : memref<0xi32>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 64512 : i32}
  }
}
