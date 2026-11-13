//===- test_sa_core_layout.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t 2> %t.err
// RUN: FileCheck %s --check-prefix=WARN --input-file=%t.err
// RUN: not aie-opt --split-input-file --aie-assign-buffer-addresses %t 2>&1 >/dev/null | FileCheck %s --check-prefix=ADDR

// The SA charges a core tile what assign-buffer-addresses' layout (layoutTile)
// cannot place, rather than its bytes padded and summed.

// A 4097-byte data region aligned to 4096 sits at 57344, and the 3000-byte
// buffer fits above it, so the tile lays out. Before, padding the region to
// 8192 bytes put the sum over 64 KB and the SA warned.
// WARN-NOT: over capacity
// ADDR-NOT: error
module @gap_fill {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<54000xi8>
    %s = aie.buffer(%p) {sym_name = "small"} : memref<3000xi8>
    aie.core(%p) { aie.end } {measured_data_size = 4097 : i32, measured_data_alignment = 4096 : i32}
  }
}

// -----

// Two 10 KB buffers pinned to bank 0 do not fit its 16 KB beside the stack,
// though the tile has room for both. Before, the sum fit and the SA was
// silent; a layout that ignored banks would be as optimistic.
// WARN: [[@LINE+5]]:3: warning: SA placer's resource estimate is over capacity
// WARN-NOT: over capacity
// ADDR: error: 'aie.buffer' op buffer "b" requires 10240 bytes in bank 0
// ADDR-NOT: error
module @bank_pinned {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %a = aie.buffer(%p) {sym_name = "a", mem_bank = 0 : i32} : memref<2560xi32>
    %b = aie.buffer(%p) {sym_name = "b", mem_bank = 0 : i32} : memref<2560xi32>
    aie.core(%p) { aie.end }
  }
}
