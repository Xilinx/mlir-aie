//===- bank_aware_reserved_data_no_free_run_error.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// See bank_aware_reserved_data_no_free_run.mlir: the stack and "a" leave a
// largest free run of exactly 0 bytes. A reservation of 1 byte fails, and the
// diagnostic reports the run as 0.

// RUN: aie-opt --verify-diagnostics --aie-assign-buffer-addresses='alloc-scheme=bank-aware' %s

module @zero_free_run_nonzero_reservation {
  aie.device(npu2) {
    // expected-warning @below {{Not all requested buffers fit in the available memory}}
    // expected-note @below {{Current configuration of buffers in bank(s)}}
    // expected-error @below {{'aie.tile' op Bank-aware allocation failed.}}
    %tile_0_2 = aie.tile(0, 2)
    // The reservation is a placeholder in the occupancy bitmap, not a buffer,
    // so it never shows up in the memory map printed alongside this. Without
    // the remark a user would see only that "a" did not fit, with no hint that
    // a one-byte reservation is what took the space it needed.
    // expected-warning @below {{Failed to allocate buffer: "a" with size: 64512 bytes}}
    // expected-remark @below {{this core reserves 1 bytes for its own data sections (reserved_data_size), placed at 0x400; 'a' would have fit without that reservation}}
    %a = aie.buffer(%tile_0_2) {sym_name = "a", aligned = false} : memref<64512xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 1 : i32}
  }
}
