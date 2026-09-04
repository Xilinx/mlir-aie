//===- bank_aware_mem_bank_pin_with_reservation_error.mlir ------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// See bank_aware_mem_bank_pin_with_reservation.mlir: "mid" pinned to bank 1
// leaves a largest free run of 40960 bytes. A reservation of 45000 bytes
// exceeds that run, so the pass reports an error and keeps both the
// requirement and the pin.
//
// The reservation is placed before any unconstrained buffer, so the error names
// the reservation itself and the only things that can be in the way at that
// point: the stack and the tile's pinned buffers.

// RUN: aie-opt --verify-diagnostics --aie-assign-buffer-addresses='alloc-scheme=bank-aware' %s

module @too_large_around_the_pin {
  aie.device(npu2) {
    // expected-error @below {{cannot reserve 45000 contiguous bytes for this core's data sections (reserved_data_size); the largest free run is 40960 bytes}}
    // expected-error @below {{'aie.tile' op Bank-aware allocation failed.}}
    %tile_0_2 = aie.tile(0, 2)
    %mid = aie.buffer(%tile_0_2) {sym_name = "mid", mem_bank = 1 : i32} : memref<8192xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 45000 : i32}
  }
}
