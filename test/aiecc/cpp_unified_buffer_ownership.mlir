//===- cpp_unified_buffer_ownership.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: peano

// A buffer's initializer belongs in exactly one core's object -- its owner's.
// AIEBufferToStandard enforces that by dropping the initializer for cores that
// do not own the buffer, keyed off the tile coordinates it is given. Unified
// lowering gives it none, so every core's module starts with every initializer
// and the globals are public, which puts them past symbol DCE. Unified has to
// re-apply ownership when it carves the device up, or each core's data section
// carries the whole device's constants.

// RUN: rm -rf %t.d && mkdir -p %t.d
// RUN: aiecc --unified --tmpdir=%t.d --get-core-elfs --verbose %s 2>&1 | FileCheck %s

// core (0,2) owns buf_a and defines it; buf_b belongs to (1,2) and must arrive
// as a declaration, exactly as the per-core path emits it.
// RUN: FileCheck --check-prefix=OWN0 %s < %t.d/peano-compat_main_core_0_2.ll
// OWN0-DAG: @buf_a = global
// OWN0-DAG: @buf_b = external global

// RUN: FileCheck --check-prefix=OWN1 %s < %t.d/peano-compat_main_core_1_2.ll
// OWN1-DAG: @buf_b = global
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
