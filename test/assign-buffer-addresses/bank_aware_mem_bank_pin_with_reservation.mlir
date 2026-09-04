//===- bank_aware_mem_bank_pin_with_reservation.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// mem_bank and reserved_data_size are both hard constraints, and this is the
// case where they conflict: a bank pin sits in the middle of the tile's free
// space, so the reservation's contiguous-run search has to account for the
// address and size of the pinned buffer, and not only for the free byte count.
//
// tile(0, 2) on npu2 has 65536 bytes across 4 banks of 16384. "mid" is pinned
// to bank 1 and lands at its start (address 16384), so the tile's free space
// splits into [1024, 16384) after the stack (15360 bytes) and [24576, 65536)
// after "mid" (40960 bytes, spanning the rest of bank 1 plus banks 2 and 3;
// addresses are linear across bank boundaries). See
// bank_aware_mem_bank_pin_with_reservation_error.mlir for a reservation that
// fits neither run.

// RUN: aie-opt --aie-assign-buffer-addresses="alloc-scheme=bank-aware" %s | FileCheck %s

// A reservation that fits only the larger [24576, 65536) run is placed there,
// and the bank pin stays where it is.
// CHECK: %mid = aie.buffer(%tile_0_2) {address = 16384 : i32, mem_bank = 1 : i32, sym_name = "mid"} : memref<8192xi8>
// CHECK: data_length = 40960 : i32, data_origin = 24576 : i32
module @fits_around_the_pin {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %mid = aie.buffer(%tile_0_2) {sym_name = "mid", mem_bank = 1 : i32} : memref<8192xi8>
    aie.core(%tile_0_2) {
      aie.end
    } {stack_size = 1024 : i32, reserved_data_size = 30000 : i32}
  }
}
