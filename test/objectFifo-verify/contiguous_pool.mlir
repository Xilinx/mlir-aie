//===- contiguous_pool.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A contiguous pool keeps its objects, and any zero objects around them, in
// one buffer.

aie.device(npu2) {
  %mem = aie.tile(0, 1)
  %b0 = aie.buffer(%mem) {sym_name = "b0"} : memref<16xi32>
  %b1 = aie.buffer(%mem) {sym_name = "b1"} : memref<16xi32>
  // expected-error@+1 {{expects one buffer holding its objects side by side}}
  aie.objectfifo.pool @p(%mem) {depth = 2 : i32, contiguous, buffers = [@b0, @b1]} : memref<8xi32> {
    aie.objectfifo.segment @s0 {offset = 0 : i32, size = 8 : i32}
  }
}

// -----

aie.device(npu2) {
  %mem = aie.tile(0, 1)
  // expected-error@+1 {{'zeroObjects' expects [before, after] counts on a contiguous pool}}
  aie.objectfifo.pool @p(%mem) {depth = 2 : i32, zeroObjects = array<i32: 1, 1>} : memref<8xi32> {
    aie.objectfifo.segment @s0 {offset = 0 : i32, size = 8 : i32}
  }
}

// -----

aie.device(npu2) {
  %mem = aie.tile(0, 1)
  // expected-error@+1 {{'zeroObjects' expects [before, after] counts on a contiguous pool}}
  aie.objectfifo.pool @p(%mem) {depth = 2 : i32, contiguous, zeroObjects = array<i32: 1>} : memref<8xi32> {
    aie.objectfifo.segment @s0 {offset = 0 : i32, size = 8 : i32}
  }
}
