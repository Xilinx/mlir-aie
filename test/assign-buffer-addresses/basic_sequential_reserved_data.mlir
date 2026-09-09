//===- basic_sequential_reserved_data.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `data_size` reaches basic-sequential too. That scheme packs every buffer
// contiguously above the stack, and the reservation joins the sequence as one
// more extent. A reservation the tile cannot hold is reported here, ahead of
// the core link's `ld.lld: section '.bss' will not fit in region 'data'`.
//
// The fallback path depends on this: a design whose bank-aware attempt fails
// for an unrelated reason retries under basic-sequential, which carries the
// reservation through.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s

// A reservation that fits leaves the layout alone: buffers pack sequentially
// from the top of the stack. tile(0, 2) on npu2 has 65536 bytes, so these two
// leave 65536 - 1024 - 32768 = 31744 free, above the 16384 reserved.
// CHECK-LABEL: module @reservation_fits
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "a"} : memref<16384xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "b"} : memref<16384xi8>
// CHECK: %core_data_0_2 = aie.buffer(%tile_0_2) {address = 33792 : i32, core_data, sym_name = "core_data_0_2"} : memref<16384xi8>
module @reservation_fits {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<16384xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<16384xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 16384 : i32}
  }
}

// -----

// Both buffers fit, but the core's own sections are a third extent, and
// 1024 + 28672 + 28672 + 16384 exceeds the tile. The allocation fails here.
module @reservation_does_not_fit {
  aie.device(npu2) {
    // expected-error @below {{allocated buffers exceeded available memory}}
    // expected-note @below {{MemoryMap}}
    // expected-error @below {{Core (0, 2) reserves 16384 bytes for its static data}}
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<28672xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<28672xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 16384 : i32}
  }
}
