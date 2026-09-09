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
    // expected-error @below {{Core (0, 2) reserves 1 bytes for its static data}}
    %tile_0_2 = aie.tile(0, 2)
    // "a" is the larger extent, so it takes every free byte and the one byte
    // the core asked for has nowhere to go. The memory map lists both, because
    // the region is a buffer like any other.
    %a = aie.buffer(%tile_0_2) {sym_name = "a", aligned = false} : memref<64512xi8>
    // expected-warning @below {{Failed to allocate this core's data sections (data_size), which needs 1 bytes}}
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, data_size = 1 : i32}
  }
}
