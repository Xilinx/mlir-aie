//===- test_sa_spill_search.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The placer weighs mem tile memory with aie-objectfifo-allocate's own buffer
// search, so the two agree on whether a placement's buffers fit, address pins
// included. A 4096-byte buffer pinned at 262144 leaves two runs of about
// 256 KB, neither of which holds the link's 300032-byte object, though the
// bytes free in all would. Pinned at 400000, it leaves one run that does.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module @fits
// CHECK: %[[M:.*]] = aie.tile(0, 1)
// CHECK: aie.buffer(%[[M]]) {sym_name = "in__cons__buf0"} : memref<300032xi8>
module @fits {
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(0, 1)
    %pin = aie.buffer(%m) {address = 400000 : i32} : memref<4096xi8>
    aie.objectfifo @in(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<300032xi8>>
    aie.objectfifo @out(%m, {%s}, 1 : i32) : !aie.objectfifo<memref<300032xi8>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
  }
}

// -----

module @does_not_fit {
  // expected-warning @below {{SA placer's resource estimate is over capacity}}
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(0, 1)
    %pin = aie.buffer(%m) {address = 262144 : i32} : memref<4096xi8>
    aie.objectfifo @in(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<300032xi8>>
    aie.objectfifo @out(%m, {%s}, 1 : i32) : !aie.objectfifo<memref<300032xi8>>
    // expected-error @below {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.link [@in] -> [@out] ([] [])
  }
}
