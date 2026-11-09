//===- test_sa_core_section_alignment.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --split-input-file --aie-place-tiles="placer=sa_placer sa-seed=1" %s -o %t
// RUN: not aie-opt --split-input-file --aie-assign-buffer-addresses %t 2>&1 >/dev/null | FileCheck %s --check-prefix=ADDR

// The SA pads a core's data region and bank reservations to the measured
// section alignment, as addressing does (CoreOp::getSectionAlignment). A
// 4097-byte region aligned to 4096 takes 8192 bytes beside a 60352-byte
// buffer and a 1 KB stack, which does not fit; with the default alignment it
// does. Before, the SA padded to the vector rule only and was silent where
// addressing failed.

// WARN: [[@LINE+4]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: {{.*}}could not be placed
// ADDR-NOT: error
module @data_align {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<60352xi8>
    aie.core(%p) { aie.end } {measured_data_size = 4097 : i32, measured_data_alignment = 4096 : i32}
  }
}

// -----

// WARN-NOT: over capacity
module @data_align_ctrl {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<60352xi8>
    aie.core(%p) { aie.end } {measured_data_size = 4097 : i32}
  }
}

// -----

// WARN: [[@LINE+4]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: {{.*}}could not be placed
// ADDR-NOT: error
module @bank_align {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<60352xi8>
    aie.core(%p) { aie.end } {measured_bank_sizes = array<i32: 0, 0, 0, 4097>, measured_bank_alignments = array<i32: 16, 16, 16, 4096>}
  }
}

// -----

// WARN-NOT: over capacity
module @bank_align_ctrl {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%p) {sym_name = "big"} : memref<60352xi8>
    aie.core(%p) { aie.end } {measured_bank_sizes = array<i32: 0, 0, 0, 4097>, measured_bank_alignments = array<i32: 16, 16, 16, 16>}
  }
}

// -----

// An existing core_data or bank_reserved buffer is charged at the alignment
// addressing places it at (bufferAlignBytes), its measured section alignment
// included. Before, the SA padded it to the vector rule only.
// WARN: [[@LINE+4]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: {{.*}}could not be placed
// ADDR-NOT: error
module @data_buffer_align {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %cd = aie.buffer(%p) {sym_name = "mine", core_data} : memref<4097xi8>
    %b = aie.buffer(%p) {sym_name = "big"} : memref<60352xi8>
    aie.core(%p) { aie.end } {measured_data_size = 4097 : i32, measured_data_alignment = 4096 : i32}
  }
}

// -----

// WARN: [[@LINE+4]]:3: warning: SA placer's resource estimate is over capacity
// ADDR: error: {{.*}}could not be placed
// ADDR-NOT: error
module @bank_buffer_align {
  aie.device(npu2_1col) {
    %p = aie.logical_tile<CoreTile>(?, ?)
    %r = aie.buffer(%p) {sym_name = "r", bank_reserved, mem_bank = 3 : i32} : memref<4097xi8>
    %b = aie.buffer(%p) {sym_name = "big"} : memref<60352xi8>
    aie.core(%p) { aie.end } {measured_bank_sizes = array<i32: 0, 0, 0, 4097>, measured_bank_alignments = array<i32: 16, 16, 16, 4096>}
  }
}
