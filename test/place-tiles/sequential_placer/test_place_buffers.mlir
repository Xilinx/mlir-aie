//===- test_place_buffers.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// Buffer fits comfortably in CoreTile L1 (npu1 has 64KB) — placer is a
// pass-through.
// CHECK-LABEL: @buffer_fits_pinned
module @buffer_fits_pinned {
  aie.device(npu1) {
    // CHECK-DAG: %[[T:.*]] = aie.tile(2, 3)
    %t = aie.logical_tile<CoreTile>(2, 3)
    %b = aie.buffer(%t) : memref<128xi64>
    aie.core(%t) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// Unconstrained CoreTile with a buffer that fits — greedy placement at (0,2).
// CHECK-LABEL: @buffer_fits_unconstrained
module @buffer_fits_unconstrained {
  aie.device(npu1) {
    // CHECK-DAG: %[[T:.*]] = aie.tile(0, 2)
    %t = aie.logical_tile<CoreTile>(?, ?)
    %b = aie.buffer(%t) : memref<1024xi32>
    aie.core(%t) { aie.end }
    // CHECK-NOT: aie.logical_tile
  }
}
