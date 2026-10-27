//===- prepare_buffers_name_taken.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Preparing buffers preserves declaration symbols, including names shared
// with the allocator's historical anonymous-name prefix.

// RUN: aie-opt --aie-prepare-buffers %s | FileCheck %s

// CHECK-DAG: aie.buffer @_anonymous0(%{{.*}}) : memref<16xi32>
// CHECK-DAG: aie.lock @_anonymous1(%{{.*}})
// CHECK-DAG: aie.buffer @unnamed(%{{.*}}) : memref<16xi32>
module {
  aie.device(npu2) {
    %t = aie.tile(0, 2)
    aie.buffer @_anonymous0(%t) : memref<16xi32>
    aie.lock @_anonymous1(%t)
    aie.buffer @unnamed(%t) : memref<16xi32>
  }
}
