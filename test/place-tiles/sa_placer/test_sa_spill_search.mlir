//===- test_sa_spill_search.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The placer weighs mem tile memory with aie-objectfifo-allocate's own buffer
// search (MemTileSpillSearch), so the two agree on whether a placement's
// buffers fit, address pins included. A 4096-byte buffer pinned at 262144
// leaves two runs of about 256 KB, neither of which holds the link's
// 300032-byte object, though the bytes free in all would. Pinned at 400000, it
// leaves one run that does.

// RUN: not aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform %s 2>/dev/null | FileCheck %s
// RUN: not aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform %s 2>&1 >/dev/null | FileCheck %s --check-prefix=DIAG

// DIAG-NOT: {{warning|error}}
// DIAG: warning: SA placer's resource estimate is over capacity
// DIAG: error: 'aie.objectfifo.pool' op could not place buffers in accessible memory with available capacity
// DIAG-NOT: {{warning|error}}

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
  aie.device(npu2_1col) {
    %s = aie.logical_tile<ShimNOCTile>(?, ?)
    %m = aie.logical_tile<MemTile>(0, 1)
    %pin = aie.buffer(%m) {address = 262144 : i32} : memref<4096xi8>
    aie.objectfifo @in(%s, {%m}, 1 : i32) : !aie.objectfifo<memref<300032xi8>>
    aie.objectfifo @out(%m, {%s}, 1 : i32) : !aie.objectfifo<memref<300032xi8>>
    aie.objectfifo.link [@in] -> [@out] ([] [])
  }
}
