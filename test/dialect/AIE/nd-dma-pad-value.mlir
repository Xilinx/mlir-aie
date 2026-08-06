//===- nd-dma-pad-value.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// The constant pad value is a per-channel property (one MemTile MM2S
// CONSTANT_PAD_VALUE register per channel), so it lives on the channel op
// (aie.dma_start / aie.dma), not on aie.dma_bd. Check the pad_value attribute
// round-trips on both channel-op forms while the pad geometry stays on the BD.

// CHECK-LABEL: module {

aie.device(xcve2802) {
  // Terminator form: aie.dma_start.
  %t1 = aie.tile(1, 1)
  %buf1 = aie.buffer(%t1) : memref<256xi32>
  aie.memtile_dma(%t1) {
    // CHECK: aie.dma_start(MM2S, 0, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}) {pad_value = 7 : i32}
    aie.dma_start("MM2S", 0, ^bd0, ^end) {pad_value = 7 : i32}
    ^bd0:
      // CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 1, const_pad_after = 1>])
      aie.dma_bd(%buf1 : memref<256xi32> offset = 0 len = 256 sizes = [2] strides = [128] pad [<const_pad_before = 1, const_pad_after = 1>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
  // Region form: aie.dma (pad_value prints via attr-dict).
  %t2 = aie.tile(2, 1)
  %buf2 = aie.buffer(%t2) : memref<256xi32>
  %pl = aie.lock(%t2, 0) {init = 1 : i32}
  %cl = aie.lock(%t2, 1) {init = 0 : i32}
  aie.memtile_dma(%t2) {
    %c1 = arith.constant 1 : i32
    // CHECK: aie.dma(MM2S, 0) {pad_value = 5 : i32} [
    %0 = aie.dma(MM2S, 0) {pad_value = 5 : i32} [{
      aie.use_lock(%cl, AcquireGreaterEqual, %c1)
      // CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 1, const_pad_after = 1>])
      aie.dma_bd(%buf2 : memref<256xi32> offset = 0 len = 256 sizes = [2] strides = [128] pad [<const_pad_before = 1, const_pad_after = 1>])
      aie.use_lock(%pl, Release, %c1)
    }]
    aie.end
  }
}
