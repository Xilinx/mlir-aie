//===- core_section_holders.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-assign-buffer-addresses %s | FileCheck %s

// An existing aie.buffer holds a core's section reservation only when it is
// the exact extent addressing would make: the same kind, bank or address, and
// size (CoreOp::getSectionReservationBuffers). Anything else is materialized
// beside it or diagnosed. Before, any buffer with the right attributes stood
// in, so a 16-byte one dropped a 12000-byte bank reservation or let buffers
// land inside a prebaked ELF's data.

// A smaller bank_reserved buffer leaves the bank's reservation to be made.
// CHECK-LABEL: module @small_bank_holder
// CHECK: sym_name = "userbuf"
// CHECK: sym_name = "bank_reserved_0_2_3"} : memref<12000xi8>
module @small_bank_holder {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %r = aie.buffer(%t) {sym_name = "userbuf", bank_reserved, mem_bank = 3 : i32} : memref<16xi8>
    aie.core(%t) { aie.end } {measured_bank_sizes = array<i32: 0, 0, 0, 12000>}
  }
}

// -----

// The exact extent holds it, and no second one is made.
// CHECK-LABEL: module @exact_bank_holder
// CHECK: sym_name = "userbuf"
// CHECK-NOT: bank_reserved_0_2_3
module @exact_bank_holder {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %r = aie.buffer(%t) {sym_name = "userbuf", bank_reserved, mem_bank = 3 : i32} : memref<12000xi8>
    aie.core(%t) { aie.end } {measured_bank_sizes = array<i32: 0, 0, 0, 12000>}
  }
}

// -----

// One buffer cannot hold two reservations.
module @one_buffer_two_roles {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    // expected-error@+1 {{cannot be both core_data and bank_reserved; a buffer holds one reservation of a core's sections}}
    %r = aie.buffer(%t) {sym_name = "both", core_data, bank_reserved, mem_bank = 3 : i32} : memref<16xi8>
    aie.core(%t) { aie.end } {data_size = 8000 : i32, measured_bank_sizes = array<i32: 0, 0, 0, 12000>}
  }
}

// -----

// A 16-byte buffer at a prebaked run's address does not cover the run, so the
// run is reserved and collides with it.
module @prebaked_small_holder {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    %r = aie.buffer(%t) {sym_name = "userbuf", bank_reserved, aligned = false, address = 4096 : i32} : memref<16xi8>
    %a = aie.buffer(%t) {sym_name = "a"} : memref<4096xi8>
    // expected-error@+1 {{would override allocated address}}
    aie.core(%t) { aie.end } {elf_file = "x.elf", measured_data_ranges = array<i32: 4096, 8192>}
  }
}

// -----

// A core_data buffer of another size than the data region is diagnosed: a
// tile has one data region.
module @core_data_wrong_size {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 2)
    // expected-error@+1 {{does not match the 8000-byte reservation core (0, 2) makes for its own sections}}
    %r = aie.buffer(%t) {sym_name = "mine", core_data} : memref<16xi8>
    aie.core(%t) { aie.end } {data_size = 8000 : i32}
  }
}
