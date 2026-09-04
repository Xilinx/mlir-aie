//===- reserved_data_size_bcf.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// BCF has no data-region concept, so reserved_data_size shows up as a
// bank-aware placement: the reservation takes the bottom of the tile and the
// three 4 KB buffers are placed above it, each marked `_reserved DMb`.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | aie-translate --aie-generate-bcf --tilecol=0 --tilerow=2 | FileCheck %s

// CHECK: _symbol a 0x7A040 4096
// CHECK: _reserved DMb 0x7A040 4096
// CHECK: _symbol b 0x7C000 4096
// CHECK: _reserved DMb 0x7C000 4096
// CHECK: _symbol c 0x7D000 4096
// CHECK: _reserved DMb 0x7D000 4096

module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<4096xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<4096xi8>
    %c = aie.buffer(%tile_0_2) {sym_name = "c"} : memref<4096xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 40000 : i32}
  }
}
