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
// An odd-sized buffer exercises this: it leaves the free run starting at a
// non-multiple of the alignment. npu2 requires 512-bit (64-byte) alignment.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck --check-prefix=SEQ %s
// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck --check-prefix=BANK %s

// 1024 (stack) + 3 (buffer) = 1027, which rounds up to 1088.
// SEQ: data_origin = 1088 : i32
// BANK: data_origin = 1088 : i32

module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %odd = aie.buffer(%t) { sym_name = "odd" } : memref<3xi8>
    %c = aie.core(%t) {
      aie.end
    }
  }
}
