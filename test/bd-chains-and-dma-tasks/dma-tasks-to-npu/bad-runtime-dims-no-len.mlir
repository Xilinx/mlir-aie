//===- bad-runtime-dims-no-len.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics --aie-dma-tasks-to-npu %s

// The whole-buffer length is only a sound default when the BD's addressing is
// constant. With runtime sizes or a runtime offset it can disagree with the
// transfer or run past the buffer, so `len` stays required. With constant
// dims it must match them, as on the static path.

aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %buf = aie.buffer(%mt) {address = 0 : i32} : memref<1024xi32>
  aie.runtime_sequence @runtime_dims(%a: i64) {
    %t = aiex.dma_configure_task(%mt, MM2S, 0) {
      // expected-error@+1 {{runtime-valued BD requires an explicit transfer length}}
      aie.dma_bd(%buf : memref<1024xi32> sizes = [%a, 16] strides = [16, 1]) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %buf = aie.buffer(%mt) {address = 0 : i32} : memref<1024xi32>
  aie.runtime_sequence @runtime_offset(%o: i32) {
    %t = aiex.dma_configure_task(%mt, MM2S, 0) {
      // expected-error@+1 {{runtime-valued BD requires an explicit transfer length}}
      aie.dma_bd(%buf : memref<1024xi32> offset = %o) {bd_id = 0 : i32}
      aie.end
    }
  }
}

// -----

// A runtime lock value alone takes the dynamic path; the whole buffer (128
// elements) is not what these dims move (64).
aie.device(npu2) {
  %mt = aie.tile(0, 1)
  %buf = aie.buffer(%mt) {address = 0 : i32} : memref<128xi32>
  %prod = aie.lock(%mt, 0) {init = 0 : i32}
  %cons = aie.lock(%mt, 1) {init = 0 : i32}
  aie.runtime_sequence @lock_dims_mismatch(%uses: i32) {
    %t = aiex.dma_configure_task(%mt, MM2S, 0) {
      aie.use_lock(%cons, AcquireGreaterEqual, %uses)
      // expected-error@+2 {{Buffer descriptor length does not match length of transfer expressed by lowest three dimensions of data layout transformation strides/wraps. BD length is 512 bytes. Lowest three dimensions of data layout transformation would result in transfer of 256 bytes.}}
      // expected-note@+1 {{Do not include the highest dimension size in transfer length, as this is the BD repeat count.}}
      aie.dma_bd(%buf : memref<128xi32> sizes = [4, 16] strides = [16, 1]) {bd_id = 0 : i32}
      aie.use_lock(%prod, Release, %uses)
      aie.end
    }
  }
}
