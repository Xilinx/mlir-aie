//===- cpp_unified_buffer_ownership.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// No core's object carries a buffer's contents, not even the object of the
// buffer's own tile. A buffer lives at an address the allocator chooses and the
// linker supplies, and its `initial_value` is device state, written by whoever
// configures the device. A definition in a core's object would place the bytes
// wherever `*(.data*)` lands, which costs the core's data region a copy it
// never reads.
//
// Unified lowering compiles every core from one llvm-linked module, so each
// core's module starts with every buffer, and the globals are public, which
// puts them past symbol DCE. A definition among them would put the whole
// device's constants in every core's data section.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --unified --tmpdir=%t.d --get-core-elfs --verbose %s 2>&1 | FileCheck %s

// RUN: FileCheck --check-prefix=OWN0 %s < %t.d/peano-compat_main_core_0_2.ll
// OWN0-DAG: @buf_a = external global
// OWN0-DAG: @buf_b = external global

// RUN: FileCheck --check-prefix=OWN1 %s < %t.d/peano-compat_main_core_1_2.ll
// OWN1-DAG: @buf_b = external global
// OWN1-DAG: @buf_a = external global

// CHECK: wrote edge

module {
  aie.device(npu2_4col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)

    %b02 = aie.buffer(%tile_0_2) { sym_name = "buf_a", initial_value = dense<11> : tensor<64xi32> } : memref<64xi32>
    %b12 = aie.buffer(%tile_1_2) { sym_name = "buf_b", initial_value = dense<22> : tensor<64xi32> } : memref<64xi32>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %v = memref.load %b02[%c0] : memref<64xi32>
      memref.store %v, %b02[%c0] : memref<64xi32>
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %v = memref.load %b12[%c0] : memref<64xi32>
      memref.store %v, %b12[%c0] : memref<64xi32>
      aie.end
    }
  }
}
