//===- test_sa_spill_stream_window.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A link whose output walks a window of its input's stream gets one pool from
// the split, its objects side by side in one buffer: here 20 objects of 5120
// bytes, 102400 bytes that spill only whole. The placer's estimate counts it
// so, and agrees with aie-objectfifo-allocate: with 110000 bytes free at home
// it fits; with 60000 free on each mem tile it fits on neither, though its
// objects alone would split across the two.

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-place-tiles='placer=sa_placer sa-seed=1' --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL: module @fits
// CHECK: %[[M0:.*]] = aie.tile(0, 1)
// CHECK: aie.buffer(%[[M0]]) {sym_name = "in__cons__buf0"} : memref<102400xi8>
module @fits {
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %r0 = aie.buffer(%m0) : memref<414288xi8>
    %r1 = aie.buffer(%m1) : memref<464288xi8>
    aie.objectfifo @in(%s0, {%m0}, [1 : i32, 20 : i32]) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo @rows(%m0 dimensionsToStream [<size = 20, stride = 5120>, <size = 2, stride = 2560>, <size = 640, stride = 1>], {%s1}, 2 : i32) : !aie.objectfifo<memref<1280xi8>>
    aie.objectfifo.link [@in] -> [@rows] ([] [])
  }
}

// -----

module @does_not_fit {
  // expected-warning @below {{SA placer's resource estimate is over capacity}}
  aie.device(npu2_2col) {
    %s0 = aie.logical_tile<ShimNOCTile>(?, ?)
    %s1 = aie.logical_tile<ShimNOCTile>(?, ?)
    %m0 = aie.logical_tile<MemTile>(0, 1)
    %m1 = aie.logical_tile<MemTile>(1, 1)
    %r0 = aie.buffer(%m0) : memref<464288xi8>
    %r1 = aie.buffer(%m1) : memref<464288xi8>
    aie.objectfifo @in(%s0, {%m0}, [1 : i32, 20 : i32]) : !aie.objectfifo<memref<5120xi8>>
    aie.objectfifo @rows(%m0 dimensionsToStream [<size = 20, stride = 5120>, <size = 2, stride = 2560>, <size = 640, stride = 1>], {%s1}, 2 : i32) : !aie.objectfifo<memref<1280xi8>>
    // expected-error @below {{could not place buffers in accessible memory with available capacity}}
    aie.objectfifo.link [@in] -> [@rows] ([] [])
  }
}
