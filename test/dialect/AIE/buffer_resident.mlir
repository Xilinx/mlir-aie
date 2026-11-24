//===- buffer_resident.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK: memref.global "private" @w : memref<16xi32>
// CHECK: aie.buffer(%{{.*}}) {resident = @w, sym_name = "buf"} : memref<16xi32>
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf", resident = @w} : memref<16xi32>
  }
}

// -----

// resident names a symbol with no memref.global at module scope.
module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{'aie.buffer' op resident names no memref.global @missing at module scope}}
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf", resident = @missing} : memref<16xi32>
  }
}

// -----

// resident's global exists, but its type does not match this buffer's.
module {
  memref.global "private" @w : memref<16xi32>
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    // expected-error@+1 {{'aie.buffer' op resident @w's type 'memref<16xi32>' does not match this buffer's 'memref<8xi32>'}}
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf", resident = @w} : memref<8xi32>
  }
}
