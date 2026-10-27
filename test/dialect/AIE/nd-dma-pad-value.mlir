//===- nd-dma-pad-value.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The constant pad value is a per-channel property (one MemTile MM2S
// CONSTANT_PAD_VALUE register per channel), so it lives on the channel op
// (aie.dma_start / aie.dma), not on aie.dma_bd. npu1 lacks the register.

// RUN: sed 's/DEVICE/npu2/g' %s | aie-opt --split-input-file | FileCheck %s
// RUN: sed 's/DEVICE/npu1/g' %s | aie-opt --split-input-file --verify-diagnostics

// Terminator form: aie.dma_start.
aie.device(DEVICE) {
  %t1 = aie.tile(1, 1)
  aie.buffer @buf1(%t1) : memref<256xi32>
  aie.memtile_dma(%t1) {
    // CHECK: aie.dma_start(MM2S, 0, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}) {pad_value = 7 : i32}
    // expected-error@+1 {{pad_value requires the CONSTANT_PAD_VALUE register}}
    aie.dma_start("MM2S", 0, ^bd0, ^end) {pad_value = 7 : i32}
    ^bd0:
      // CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 1, const_pad_after = 1>])
      %buf1 = aie.get_buffer @buf1 : memref<256xi32>
      aie.dma_bd(%buf1 : memref<256xi32> offset = 0 len = 256 sizes = [2] strides = [128] pad [<const_pad_before = 1, const_pad_after = 1>])
      aie.next_bd ^end
    ^end:
      aie.end
  }
}

// -----

// Region form: aie.dma (pad_value prints via attr-dict).
aie.device(DEVICE) {
  %t2 = aie.tile(2, 1)
  aie.buffer @buf2(%t2) : memref<256xi32>
  aie.lock @pl(%t2, 0) {init = 1 : i32}
  aie.lock @cl(%t2, 1) {init = 0 : i32}
  aie.memtile_dma(%t2) {
    %c1 = arith.constant 1 : i32
    // CHECK: aie.dma(MM2S, 0) {pad_value = 5 : i32} [
    // expected-error@+1 {{pad_value requires the CONSTANT_PAD_VALUE register}}
    aie.dma(MM2S, 0) {pad_value = 5 : i32} [{
      aie.use_lock(@cl, AcquireGreaterEqual, %c1)
      // CHECK: aie.dma_bd({{.*}} pad [<const_pad_before = 1, const_pad_after = 1>])
      %buf2 = aie.get_buffer @buf2 : memref<256xi32>
      aie.dma_bd(%buf2 : memref<256xi32> offset = 0 len = 256 sizes = [2] strides = [128] pad [<const_pad_before = 1, const_pad_after = 1>])
      aie.use_lock(@pl, Release, %c1)
    }]
    aie.end
  }
}
