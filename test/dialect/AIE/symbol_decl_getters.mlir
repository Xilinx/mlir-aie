//===- symbol_decl_getters.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file %s | FileCheck %s

module {
  aie.device(npu1) {
    %tile = aie.tile(1, 2)
    aie.lock_decl @l0(%tile, 0) {init = 1 : i32}
    %l = aie.get_lock @l0

    aie.buffer_decl @b0(%tile) : memref<16xi32>
    %b = aie.get_buffer @b0 : memref<16xi32>

    aie.external_buffer_decl @eb0 : memref<32xi32>
    %eb = aie.get_external_buffer @eb0 : memref<32xi32>
  }
}
// CHECK: aie.lock_decl @l0
// CHECK: %[[L:.*]] = aie.get_lock @l0
// CHECK: aie.buffer_decl @b0
// CHECK: %[[B:.*]] = aie.get_buffer @b0 : memref<16xi32>
// CHECK: aie.external_buffer_decl @eb0
// CHECK: %[[EB:.*]] = aie.get_external_buffer @eb0 : memref<32xi32>
