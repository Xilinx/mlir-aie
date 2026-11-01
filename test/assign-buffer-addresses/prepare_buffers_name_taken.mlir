//===- prepare_buffers_name_taken.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A buffer or lock name is not a Symbol, yet naming an unnamed buffer must
// still skip it.

// RUN: aie-opt --aie-prepare-buffers %s | FileCheck %s

// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "_anonymous0"} : memref<16xi32>
// CHECK-DAG: aie.lock(%{{.*}}) {sym_name = "_anonymous1"}
// CHECK-DAG: aie.buffer(%{{.*}}) {sym_name = "_anonymous2"} : memref<16xi32>
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    %named = aie.buffer(%t) {sym_name = "_anonymous0"} : memref<16xi32>
    %lock = aie.lock(%t) {sym_name = "_anonymous1"}
    %unnamed = aie.buffer(%t) : memref<16xi32>
  }
}
