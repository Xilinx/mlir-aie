//===- nd-dma-pad-value.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// The constant pad value is a per-channel property (one MemTile MM2S
// CONSTANT_PAD_VALUE register per channel), so it lives on the channel op
// (aie.dma_start), not on aie.dma_bd. Check the pad_value attribute round-trips
// on the channel op while the pad geometry stays on the BD.

// CHECK-LABEL: module {

aie.device(xcve2802) {
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
}
