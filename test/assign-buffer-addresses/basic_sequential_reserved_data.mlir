//===- basic_sequential_reserved_data.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// `reserved_data_size` constrains basic-sequential too. That scheme packs every
// buffer contiguously above the stack, so its free run is the space above the
// last buffer. A reservation larger than that run is reported here, and not at
// the core link as `ld.lld: section '.bss' will not fit in region 'data'`.
//
// The fallback path depends on this check: a design whose bank-aware attempt
// fails for an unrelated reason retries under basic-sequential, which carries
// the reservation through.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses="alloc-scheme=basic-sequential" %s | FileCheck %s

// A reservation that fits leaves the layout alone: buffers pack sequentially
// from the top of the stack. tile(0, 2) on npu2 has 65536 bytes, so these two
// leave 65536 - 1024 - 32768 = 31744 free, above the 16384 requested.
// CHECK-LABEL: module @reservation_fits
// CHECK: %a = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "a"} : memref<16384xi8>
// CHECK: %b = aie.buffer(%tile_0_2) {address = 17408 : i32, sym_name = "b"} : memref<16384xi8>
// The granted region the linker script receives. It is the run that survives
// placement, so it is at least reserved_data_size.
// CHECK: data_length = 31744 : i32, data_origin = 33792 : i32
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
// above them, less than the core's own sections need. The allocation fails
// here.
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
