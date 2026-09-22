// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt --aie-assign-buffer-addresses --split-input-file %s | FileCheck %s

// Section alignment, not just vector alignment, governs an exact reservation.
// CHECK: address = 1280 : i32, core_data, mem_bank = 0 : i32, sym_name = "core_data_0_2"
// CHECK-SAME: memref<64xi8>
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %odd = aie.buffer(%t) {sym_name = "odd", address = 1024 : i32} : memref<3xi8>
    aie.core(%t) {
      aie.end
    } {data_size = 64 : i32, measured_data_size = 64 : i32, measured_data_alignment = 256 : i32}
  }
}

// -----

// An inferred reservation has the same alignment, without increasing its size.
// CHECK: address = 1280 : i32, core_data, mem_bank = 0 : i32, sym_name = "core_data_0_2"
// CHECK-SAME: memref<64xi8>
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %odd = aie.buffer(%t) {sym_name = "odd", address = 1024 : i32} : memref<3xi8>
    aie.core(%t) {
      aie.end
    } {measured_data_size = 64 : i32, measured_data_alignment = 256 : i32}
  }
}

// -----

// Bank-pinned sections have their own alignment, independent of generic data.
// CHECK: address = 1280 : i32, bank_reserved, mem_bank = 0 : i32, sym_name = "bank_reserved_0_2_0"
// CHECK-SAME: memref<64xi8>
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %odd = aie.buffer(%t) {sym_name = "odd", address = 1024 : i32} : memref<3xi8>
    %tail = aie.buffer(%t) {sym_name = "tail", address = 1344 : i32} : memref<15040xi8>
    aie.core(%t) {
      aie.end
    } {measured_bank_sizes = array<i32: 64, 0, 0, 0>, measured_bank_alignments = array<i32: 256, 1, 1, 1>}
  }
}

// -----

// A bank reservation whose size equals its alignment still needs an aligned
// start; rounding the size alone would leave it at 1024.
// CHECK: address = 4096 : i32, bank_reserved, mem_bank = 0 : i32, sym_name = "bank_reserved_0_2_0"
// CHECK-SAME: memref<4096xi8>
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.core(%t) {
      aie.end
    } {measured_bank_sizes = array<i32: 4096, 0, 0, 0>, measured_bank_alignments = array<i32: 4096, 1, 1, 1>}
  }
}
