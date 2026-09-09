//===- data_region_alignment.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The core's data region starts aligned, whatever address the buffers above it
// end on. The linker starts .data at a multiple of its strongest section
// alignment, so an unaligned ORIGIN loses that much of the region to padding,
// which is enough to overflow a reservation of the exact size.
//
// An odd-sized buffer pinned at the bottom exercises this: it ends at 1027, a
// non-multiple of the alignment. npu2 requires 512-bit (64-byte) alignment.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck --check-prefix=SEQ %s
// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck --check-prefix=BANK %s

// 1024 (stack) + 3 (buffer) = 1027, which rounds up to 1088.
// SEQ: %core_data_0_2 = aie.buffer(%tile_0_2) {address = 1088 : i32, core_data, sym_name = "core_data_0_2"}
// BANK: %core_data_0_2 = aie.buffer(%tile_0_2) {address = 1088 : i32, core_data, mem_bank = 0 : i32, sym_name = "core_data_0_2"}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %odd = aie.buffer(%t) { sym_name = "odd", address = 1024 : i32 } : memref<3xi8>
    %c = aie.core(%t) {
      aie.end
    } { data_size = 4096 : i32 }
  }
}
