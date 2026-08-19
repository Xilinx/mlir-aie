//===- basic_sequential_reserved_data.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `reserved_data_size` is not a bank-aware-only constraint. Basic-sequential
// packs every buffer contiguously above the stack, so its free run is whatever
// sits above the last buffer -- but a reservation that does not fit there is
// just as unlinkable, and must be reported here rather than escaping to the
// core link as an `ld.lld: section '.bss' will not fit in region 'data'`.
//
// This matters most on the fallback path: a design whose bank-aware attempt
// fails for an unrelated reason retries under basic-sequential, and without
// this check the reservation would be silently dropped on the way through.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s

// A reservation that fits is not disturbed: buffers still pack sequentially
// from the top of the stack. tile(0, 2) on npu2 has 65536 bytes, so these two
// leave 65536 - 1024 - 32768 = 31744 free, comfortably above the 16384 asked
// for.
// CHECK-LABEL: module @reservation_fits
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "a"} : memref<16384xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "b"} : memref<16384xi8>
module @reservation_fits {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<16384xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<16384xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 16384 : i32}
  }
}

// -----

// Every buffer fits, but they leave only 65536 - 1024 - 57344 = 7168 bytes
// above them -- less than the core's own sections need. That is a failure of
// the allocation, not of the link.
module @reservation_does_not_fit {
  aie.device(npu2) {
    // expected-warning @below {{buffers leave only 7168 contiguous bytes for the core's data sections, which need 16384 bytes}}
    // expected-error @below {{'aie.tile' op Basic sequential allocation failed.}}
    %tile_0_2 = aie.tile(0, 2)
    %a = aie.buffer(%tile_0_2) {sym_name = "a"} : memref<28672xi8>
    %b = aie.buffer(%tile_0_2) {sym_name = "b"} : memref<28672xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 16384 : i32}
  }
}
